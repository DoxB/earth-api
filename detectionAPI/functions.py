import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
import time

import torch
import torch.nn.functional as F
import numpy as np
from rvgae_model import RVGAE
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URL = os.environ.get('NEO4J_URL')
NEO4J_PORT = os.environ.get('NEO4J_PORT')
NEO4J_ID = os.environ.get('NEO4J_ID')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')

class SelectNeo4j:
    @staticmethod
    def process_neo4j_to_indexed_format(query_results):
        """
        Neo4j 쿼리 결과를 처리하여:
        1. 노드들을 0번부터 인덱싱하여 딕셔너리로 저장
        2. sup, r, sub 관계를 [sup_idx, rel_idx, sub_idx] 형태로 변환
        """

        # 노드와 관계 수집
        nodes = {}  # name -> node_info 매핑
        relations = set()  # 고유 관계 타입들
        triples = []  # (sup_name, rel_type, sub_name) 튜플들

        # 쿼리 결과 처리
        for record in query_results:
            sup = record['sup']
            rel = record['r']
            sub = record['sub']

            # 노드 정보 추출
            sup_name = sup.get('name', f"node_{sup.id}")
            sub_name = sub.get('name', f"node_{sub.id}")
            rel_type = rel.type

            # 노드 정보 저장
            nodes[sup_name] = {
                'name': sup_name,
                'embedding': sup.get('embedding', []),
                'createdTimestamp': sup.get('createdTimestamp', 0),
                'oriTopic': sup.get('oriTopic', ''),
                'neo4j_id': sup.id
            }

            nodes[sub_name] = {
                'name': sub_name,
                'embedding': sub.get('embedding', []),
                'createdTimestamp': sub.get('createdTimestamp', 0),
                'oriTopic': sub.get('oriTopic', ''),
                'neo4j_id': sub.id
            }

            # 관계와 트리플 저장
            relations.add(rel_type)
            triples.append((sup_name, rel_type, sub_name))

        # 노드들을 생성시간 순으로 정렬하여 0부터 순서대로 인덱스 부여
        sorted_node_names = sorted(nodes.keys(),
                                key=lambda x: nodes[x]['createdTimestamp'],
                                reverse=False)

        # 노드 이름 -> 사용자 정의 인덱스 매핑 (0부터 시작)
        name_to_idx = {name: idx for idx, name in enumerate(sorted_node_names)}

        # 관계 타입 -> 고정 인덱스 매핑
        rel_to_idx = {
            'isCauseOf': 0,
            'isGeneralOf': 1
        }

        # 사용자 정의 인덱싱된 노드 딕셔너리 생성 (0부터 시작하는 인덱스 -> 노드 정보)
        indexed_nodes = {}
        for idx, name in enumerate(sorted_node_names):
            indexed_nodes[idx] = nodes[name].copy()
            indexed_nodes[idx]['custom_index'] = idx  # 사용자 정의 인덱스
            indexed_nodes[idx]['original_neo4j_id'] = nodes[name]['neo4j_id']  # 원본 Neo4j ID는 참조용으로 보관

        # 관계를 인덱스 형태로 변환
        indexed_relations = []
        for sup_name, rel_type, sub_name in triples:
            sup_idx = name_to_idx[sup_name]
            rel_idx = rel_to_idx[rel_type]
            sub_idx = name_to_idx[sub_name]
            indexed_relations.append([sup_idx, rel_idx, sub_idx])

            # indexed_nodes: {0: {node_info}, 1: {node_info}, ...} - 사용자 정의 0부터 시작
            # indexed_relations: [[0, 0, 1], [1, 1, 2], ...] - 사용자 정의 인덱스 사용 [start, rel, end]
        return indexed_nodes, indexed_relations

    def extract_topic_subgraph_indexed(neo4j_uri=f"{NEO4J_URL}:{NEO4J_PORT}",
                                  username=NEO4J_ID,
                                  password=NEO4J_PASSWORD):
        """
        Neo4j에서 최신 노드의 oriTopic 기반으로 관계를 추출하고 인덱스 형태로 변환
        """

        query = """
            MATCH (latest:Youtube)
            WITH latest ORDER BY latest.createdTimestamp DESC LIMIT 1
            WITH latest.oriTopic as targetTopic
            MATCH (sup:Youtube)-[r]->(sub:Youtube)
            WHERE sup.oriTopic = targetTopic AND sub.oriTopic = targetTopic
            AND type(r) IN ['isCauseOf', 'isGeneralOf']
            RETURN sup, r, sub
        """

        driver = GraphDatabase.driver(neo4j_uri, auth=(username, password))

        try:
            with driver.session() as session:
                result = list(session.run(query))
                node_dict, relations = SelectNeo4j.process_neo4j_to_indexed_format(result)
                return node_dict, relations
        finally:
            driver.close()

class PreprocessDatasets:
    def extract_embedding(node_dict):
        emb_list = []
        for _, v in node_dict.items():
            emb_list.append(v['embedding'])
        return emb_list

    def convert_relation(relations):
        edge_list = []
        start = []
        end = []
        labels = []

        for rel in relations:
            start.append(rel[0])
            end.append(rel[2])
            labels.append(rel[1])

        edge_list.append(start)
        edge_list.append(end)

        return edge_list, labels

    def filter_direct(node_dict, pred_list):
        pred_rel = []
        for r in pred_list:
            if r[3] == 0:
                rel = '인과'
            else:
                rel = '관계있음'
            pred_rel.append([node_dict[r[0]]['name'], node_dict[r[1]]['name'], rel, r[2]])

        rel_score = dict()
        for k in pred_rel:
            rel_score[(k[0], k[1])] = [k[2], k[3]]

        tmp = list(rel_score.keys())  # 키 목록을 리스트로 복사
        for h in tmp:
            if (h[1], h[0]) in rel_score and h in rel_score:
                if rel_score[h][1] >= rel_score[(h[1], h[0])][1]:
                    rel_score.pop((h[1], h[0]), None)
                else:
                    rel_score.pop(h, None)

        filter_results = []
        for k, v in rel_score.items():
            filter_results.append([k[0], k[1], v[0]])

        return filter_results


class DetectHidedRelation:
    def predict(emb_list, edge_list, labels, seed=42):

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        x = torch.tensor(emb_list, dtype=torch.float)
        pos_edge_index = torch.tensor(edge_list, dtype=torch.long)
        edge_type = torch.tensor(labels, dtype=torch.long)

        edge_index = pos_edge_index
        in_channels = x.size(1)
        hidden_channels = 64
        out_channels = 32
        num_relations = int(edge_type.max().item()) + 1

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RVGAE(in_channels, hidden_channels, out_channels, num_relations).to(device)

        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_type = edge_type.to(device)
        pos_edge_index = pos_edge_index.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # === edge_type 기반 가중치 자동 계산 ===
        with torch.no_grad():
            edge_type_counts = torch.bincount(edge_type, minlength=num_relations).float()
            class_weights = 1.0 / (torch.log1p(edge_type_counts) + 1e-6)
            class_weights = class_weights / class_weights.sum()
            class_weights = class_weights.to(device)

        # === 학습 루프 ===
        epochs = 100
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            _,_, mean, logstd, z = model(x, edge_index, edge_type, pos_edge_index)
            pos_out,_ = model.decode(z, pos_edge_index)

            pos_loss = F.binary_cross_entropy(pos_out, torch.ones_like(pos_out))

            kl_loss = -0.5 / x.size(0) * torch.mean(torch.sum(
                1 + 2 * logstd - mean**2 - torch.exp(2 * logstd), dim=1))

            # === 링크 타입 분류 손실 ===
            _, type_pred = model.decode(z, edge_index)
            type_loss = F.cross_entropy(type_pred, edge_type, weight=class_weights)

            loss = pos_loss + kl_loss + type_loss
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        # === 숨은 링크 + 타입 예측 ===
        model.eval()
        with torch.no_grad():
            _, _, _, _, z = model(x, edge_index, edge_type, pos_edge_index)

            num_nodes = x.size(0)
            all_pairs = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]

            gt_pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist())).union(set(zip(edge_index[1].tolist(), edge_index[0].tolist())))

            predict_pairs = [pair for pair in all_pairs if pair not in gt_pairs]
            batch_size = 100000
            results = []

            for i in range(0, len(predict_pairs), batch_size):
                batch = predict_pairs[i:i + batch_size]
                edge_batch = torch.tensor(batch, dtype=torch.long).t().to(device)  # shape: (2, B)

                link_pred, type_pred = model.decode(z, edge_batch)
                scores = link_pred.cpu().numpy()
                pred_types = torch.argmax(type_pred, dim=1).cpu().numpy()

                for (n1, n2), s, rel in zip(batch, scores, pred_types):
                    if s >= 0.995: # threshold
                        results.append((int(n1), int(n2), float(s), int(rel)))  # only high-score predictions

        return results

class UpdataNeo4j:
    def update_pred_neo4j(pred_list):
        url = f"{NEO4J_URL}:{NEO4J_PORT}"
        auth = (NEO4J_ID, NEO4J_PASSWORD)

        with GraphDatabase.driver(url, auth=auth) as driver:
            driver.verify_connectivity()
            with driver.session(database="neo4j") as session:
                # 노드 생성
                gen_pred_rel = '''
                    UNWIND $node_list AS row
                    WITH row[0] AS source_name, row[1] AS target_name, row[2] AS relation_type

                    MATCH (source {name: source_name}), (target {name: target_name})
                    WHERE NOT EXISTS((source)-[]->(target)) AND NOT EXISTS((target)-[]->(source))

                    FOREACH (x IN CASE WHEN relation_type = "관계있음" THEN [1] ELSE [] END |
                        CREATE (source)-[:isGeneralOf_pred]->(target)
                    )
                    FOREACH (x IN CASE WHEN relation_type = "인과" THEN [1] ELSE [] END |
                        CREATE (source)-[:isCauseOf_pred]->(target)
                    )
                '''
                session.run(gen_pred_rel, node_list=pred_list)

        return "ok"