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
    node_dict, relations = SelectNeo4j.extract_topic_subgraph_indexed()
    emb_list = PreprocessDatasets.extract_embedding(node_dict)
    edge_list, labels = PreprocessDatasets.convert_relation(relations)
    results = DetectHidedRelation.predict(emb_list, edge_list, labels)
    filter_results = PreprocessDatasets.filter_direct(node_dict, results)
    # UpdataNeo4j.update_pred_neo4j(filter_results)
    print(filter_results)

    answer = []
    for source, target, rel_type  in filter_results:
        tmp = {}
        tmp['source'] = source
        tmp['target'] = target
        tmp['rel_type'] = rel_type
        answer.append(tmp)
    # return {"result": len(filter_results), "node": len(emb_list), 'filter_results':filter_results}
    return answer