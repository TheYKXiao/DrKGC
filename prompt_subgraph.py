import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import networkx as nx
import pickle
import argparse


def add_prompt(raw, relation_questions_A_to_B, relation_questions_B_to_A, bkg):
    rel = raw['triple'][1]
    query_entity = raw['query_entity']
    rank_entities = raw['rank_entities']
    pred_type = raw['type']
    answer_options = "(" + ", ".join([f"'{name}'" for name in rank_entities]) + ")"
    refer_parts = []
    refer_parts.append(f"'{query_entity}': [QUERY]")
    for name in rank_entities:
        refer_parts.append(f"'{name}': [ENTITY]")
    if len(refer_parts) > 2:
        refer_str = ", ".join(refer_parts[:2]) + "," + ", ".join(refer_parts[2:])
    else:
        refer_str = ", ".join(refer_parts)
    if pred_type == "predicted_tail":
        question_template = relation_questions_A_to_B.get(rel, "What is related to {}?")
    elif pred_type == "predicted_head":
        question_template = relation_questions_B_to_A.get(rel, "What is related to {}?")
    question = question_template.format(query_entity)

    if bkg:
        prompt = ("You are a biomedical scientist. The task is to predict the answer based on the given question, and you only need to answer one entity. The answer must be in " + answer_options + ".\nYou can refer to the entity embeddings: " + refer_str + ".\n\nQuestion: " + question + "\nAnswer: ")
    else:
        prompt = ("You are an excellent linguist. The task is to predict the answer based on the given question, and you only need to answer one entity. The answer must be in " + answer_options + ".\nYou can refer to the entity embeddings: " + refer_str + ".\n\nQuestion: " + question + "\nAnswer: ")

    if pred_type == "predicted_tail":
        answer = raw['triple'][2]
    elif pred_type == "predicted_head":
        answer = raw['triple'][0]
    raw['input'] = prompt
    raw['output'] = answer


def map_graph(df, entity2id, relation2id):
    df_mapped = df.copy()
    df_mapped[0] = df[0].map(entity2id)
    df_mapped[1] = df[1].map(relation2id)
    df_mapped[2] = df[2].map(entity2id)
    return df_mapped


def process_key_value(key, value):
    processed_value = [
        [item.lstrip('_') for item in sublist]
        for sublist in value
    ]
    unique_value = [list(x) for x in set(tuple(sublist) for sublist in processed_value)]
    return key, unique_value


def subgraph_func(test_1, graph_size, G, rules):
    test_subgraph = []
    sizes = []
    exps = []

    def apply_rule_sequence(start_node, rule_tuple, target_node, G):
        temp_path = []
        current_node = start_node
        for step_relation in rule_tuple:
            neighbors = G[current_node]
            found_next = False
            for nb_node, edges_dict in neighbors.items():
                for edge_key, edge_data in edges_dict.items():
                    if edge_data.get('relation') == step_relation:
                        temp_path.append([current_node, step_relation, nb_node])
                        current_node = nb_node
                        found_next = True
                        break
                if found_next:
                    break
            if not found_next:
                return False, []
        if current_node == target_node:
            return True, temp_path
        else:
            return False, []
    for item in tqdm(test_1):
        rel_id = item['triple_id'][1]
        pred_type = item['type']
        triple_id = item['triple_id']
        query_entity_id = item['query_entity_id']
        rank_entites_id = item['rank_entities_id']
        subg = []
        exp = 0

        # Shortest path.
        for node in rank_entites_id:
            node = int(node)
            try:
                path = nx.shortest_path(G, source=node, target=query_entity_id)
                for i in range(len(path) - 1):
                    src = int(path[i])
                    dst = int(path[i + 1])
                    relations = list(G[src][dst].keys())
                    relation = int(G[src][dst][relations[0]]['relation'])
                    subg.append([src, relation, dst])
            except:
                exp += 1
                continue
        
        # Rule-based paths (query entity to candidate).
        if len(subg) < graph_size:
            rule_sequences = rules.get(rel_id, [])
            for rank_ent in rank_entites_id:
                if len(subg) >= graph_size:
                    break
                for rule_tuple in rule_sequences:
                    if len(subg) >= graph_size:
                        break
                    success, temp_path = apply_rule_sequence(
                        start_node=query_entity_id,
                        rule_tuple=rule_tuple,
                        target_node=rank_ent,
                        G=G
                    )
                    if success:
                        for triple in temp_path:
                            subg.append(triple)
                            if len(subg) >= graph_size:
                                break
        
        # Rule-based paths (query entity or candidate to other entity).
        if len(subg) < graph_size:
            rule_sequences = rules.get(rel_id, [])
            entity_list_to_try = [query_entity_id] + list(rank_entites_id)
            for current_entity in entity_list_to_try:
                if len(subg) >= graph_size:
                    break
                for rule_tuple in rule_sequences:
                    if len(subg) >= graph_size:
                        break
                    temp_path = []
                    success = True
                    start_node = current_entity
                    for step_relation in rule_tuple:
                        neighbors = G[start_node]
                        next_node = None
                        for nb_node, edges_dict in neighbors.items():
                            found = False
                            for edge_key, edge_data in edges_dict.items():
                                if edge_data.get('relation') == step_relation:
                                    next_node = nb_node
                                    temp_path.append([start_node, step_relation, next_node])
                                    found = True
                                    break
                            if found:
                                break
                        if next_node is None:
                            success = False
                            break
                        else:
                            start_node = next_node
                    if success:
                        for triple in temp_path:
                            subg.append(triple)
                            if len(subg) >= graph_size:
                                break
        
        # Suggraph size control.
        if len(subg) < graph_size:
            allowed_entities = set([query_entity_id] + list(rank_entites_id))
            possible_triples = []
            for src in allowed_entities:
                for dst in allowed_entities:
                    if src == dst:
                        continue
                    if G.has_edge(src, dst):
                        for key, edge_data in G[src][dst].items():
                            relation = edge_data['relation']
                            possible_triples.append([src, relation, dst])
            existing_triples = set(tuple(triple) for triple in subg)
            possible_triples = [
                triple for triple in possible_triples
                if tuple(triple) not in existing_triples
            ]
            for triple in possible_triples:
                if len(subg) >= graph_size:
                    break
                subg.append(triple)

        exps.append(exp)
        sizes.append(len(subg))
        test_subgraph.append(subg)

    return test_subgraph


def default(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def main(args):
    with open(args.train_json_path, 'r', encoding='utf-8') as json_file:
        train_json = json.load(json_file)
    with open(args.valid_json_path, 'r', encoding='utf-8') as json_file:
        valid_json = json.load(json_file)
    with open(args.test_json_path, 'r', encoding='utf-8') as json_file:
        test_json = json.load(json_file)
    with open(args.tail_pred_lex, 'r', encoding='utf-8') as json_file:
        relation_questions_A_to_B = json.load(json_file)
    with open(args.head_pred_lex, 'r', encoding='utf-8') as json_file:
        relation_questions_B_to_A = json.load(json_file)

    with open(args.entity2id_path, 'rb') as file:
        entity2id = pickle.load(file)
    with open(args.id2entity_path, 'rb') as file:
        id2entity = pickle.load(file)
    with open(args.id2relation_path, 'rb') as file:
        id2relation = pickle.load(file)
    relation2id = {v: k for k, v in id2relation.items()}

    train_df = pd.read_csv(args.train_raw, sep="\t", header=None)
    valid_df = pd.read_csv(args.valid_raw, sep="\t", header=None)
    test_df = pd.read_csv(args.test_raw, sep="\t", header=None)
    graph_df = pd.concat([train_df, valid_df, test_df])

    train_id = map_graph(train_df, entity2id, relation2id)
    valid_id = map_graph(valid_df, entity2id, relation2id)
    test_id = map_graph(test_df, entity2id, relation2id)
    graph_id = map_graph(graph_df, entity2id, relation2id)
    tv_id = pd.concat([train_id, valid_id])

    # Add prompts
    for raw in train_json:
        add_prompt(raw, relation_questions_A_to_B, relation_questions_B_to_A, args.bkg)
    for raw in valid_json:
        add_prompt(raw, relation_questions_A_to_B, relation_questions_B_to_A, args.bkg)
    for raw in test_json:
        add_prompt(raw, relation_questions_A_to_B, relation_questions_B_to_A, args.bkg)

    with open(args.rules_path, 'r', encoding='utf-8') as json_file:
        rules_name = json.load(json_file)
    rules_name_1 = {}
    for key, value in rules_name.items():
        new_key, new_value = process_key_value(key, value)
        rules_name_1[new_key] = new_value

    rules = {
        relation2id.get(key, 'Unknown'): [
            [relation2id.get(item, 'Unknown') for item in rule] for rule in value
        ]
        for key, value in rules_name_1.items()
    }

    G = nx.MultiGraph()
    for index, row in tv_id.iterrows():
        head = int(row[0])
        relation = row[1]
        tail = int(row[2])
        G.add_edge(head, tail, relation=relation)

    # Retrieve subgraph
    train_subgraph = subgraph_func(train_json, args.graph_size, G, rules)
    for i in range(len(train_subgraph)):
        train_json[i]['subgraph'] = train_subgraph[i]

    valid_subgraph = subgraph_func(valid_json, args.graph_size, G, rules)
    for i in range(len(valid_subgraph)):
        valid_json[i]['subgraph'] = valid_subgraph[i]

    test_subgraph = subgraph_func(test_json, args.graph_size, G, rules)
    for i in range(len(test_subgraph)):
        test_json[i]['subgraph'] = test_subgraph[i]

    with open(args.valid_path_saved, 'w', encoding='utf-8') as f:
        json.dump(valid_json, f, ensure_ascii=False, indent=4, default=default)
    with open(args.train_path_saved, 'w', encoding='utf-8') as f:
        json.dump(train_json, f, ensure_ascii=False, indent=4, default=default)
    with open(args.test_path_saved, 'w', encoding='utf-8') as f:
        json.dump(test_json, f, ensure_ascii=False, indent=4, default=default)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_raw", required=True, help="Raw KG training set (triple tsv).")
    parser.add_argument("--valid_raw", required=True, help="Raw KG validation set (triple tsv).")
    parser.add_argument("--test_raw", required=True, help="Raw KG testing set (triple tsv).")
    parser.add_argument("--entity2id_path", required=True, help="Entity mapping file.")
    parser.add_argument("--id2entity_path", required=True, help="Entity mapping file.")
    parser.add_argument("--id2relation_path", required=True, help="Relation mapping file.")
    parser.add_argument("--train_json_path", required=True, help="Coarse ranks from lightweight model on splited raw validation set.")
    parser.add_argument("--valid_json_path", required=True, help="Coarse ranks from lightweight model on splited raw validation set.")
    parser.add_argument("--test_json_path", required=True, help="Coarse ranks from lightweight model on raw testing set.")
    parser.add_argument("--train_path_saved", required=True, help="Training set for LLM finetuning.")
    parser.add_argument("--valid_path_saved", required=True, help="Validation set for LLM finetuning.")
    parser.add_argument("--test_path_saved", required=True, help="Testing set for finetuned LLM.")
    parser.add_argument("--tail_pred_lex", required=True, help="Tail prediction lexicon file.")
    parser.add_argument("--head_pred_lex", required=True, help="Head prediction lexicon file.")
    parser.add_argument("--rules_path", required=True, help="Logic rule file for KG.")
    parser.add_argument("--graph_size", type=int, required=True, help="Tau, the limitation of graph size.")
    parser.add_argument("--bkg", action="store_true", help="Is it a biomedical knowledge graph?")
    args = parser.parse_args()

    main(args)




