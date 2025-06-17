from fastapi import FastAPI
from functions import *
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# origins에는 protocal, domain, port만 등록한다.
origins = [
    # "http://192.168.0.13:3000", # url을 등록해도 되고
    "*" # private 영역에서 사용한다면 *로 모든 접근을 허용할 수 있다.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, # cookie 포함 여부를 설정한다. 기본은 False
    allow_methods=["*"],    # 허용할 method를 설정할 수 있으며, 기본값은 'GET'이다.
    allow_headers=["*"],	# 허용할 http header 목록을 설정할 수 있으며 Content-Type, Accept, Accept-Language, Content-Language은 항상 허용된다.
)

@app.get("/generate_hiding_relation")
def generate_hiding_relation():
    """
    숨겨진 링크 및 관계 예측 API

    Neo4j에서 추출된 최신 토픽 기반 서브그래프를 사용하여 RVGAE 모델로
    관측되지 않은 노드 쌍 간의 관계를 예측하고, 고신뢰(high-confidence) 예측 결과를 반환합니다.

    Returns:
        List[dict]: 예측된 관계 리스트. 각 항목은 {
            "source": str,      # 출발 노드 이름
            "target": str,      # 도착 노드 이름
            "rel_type": str     # 예측된 관계 타입 ("인과" 또는 "관계있음")
        }
    """
    # 1. Neo4j에서 최신 주제 토픽에 해당하는 관계 데이터 추출
    node_dict, relations = SelectNeo4j.extract_topic_subgraph_indexed()

    # 2. 노드 임베딩 및 관계 형태 변환
    emb_list = PreprocessDatasets.extract_embedding(node_dict)
    edge_list, labels = PreprocessDatasets.convert_relation(relations)

    # 3. 숨겨진 링크 예측 수행 (RVGAE)
    results = DetectHidedRelation.predict(emb_list, edge_list, labels)

    # 4. 예측된 관계 중 중복 제거 및 방향성 필터링
    filter_results = PreprocessDatasets.filter_direct(node_dict, results)

    # 5. JSON 응답 포맷 구성
    answer = []
    for source, target, rel_type in filter_results:
        answer.append({
            "source": source,
            "target": target,
            "rel_type": rel_type
        })
    return answer