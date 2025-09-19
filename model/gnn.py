import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import dgl
from dgl.nn import RelGraphConv
from transformers import activations

__all__ = ["GNN", "GraphEnhancer"] 

class GNN(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, num_rels, num_hidden_layers=1):
        super().__init__()
        self.layer1 = RelGraphConv(in_dim, h_dim, num_rels, regularizer=None, num_bases=None)
        self.hidden_layers = nn.ModuleList([
            RelGraphConv(h_dim, h_dim, num_rels, regularizer=None, num_bases=None)
            for _ in range(num_hidden_layers - 1)
        ])
        self.layer_last = RelGraphConv(h_dim, out_dim, num_rels, regularizer=None, num_bases=None)

    def forward(self, g, feats, etypes):
        x = self.layer1(g, feats, etypes)
        x = F.relu(x)
        for layer in self.hidden_layers:
            x = layer(g, x, etypes)
            x = F.relu(x)
        x = self.layer_last(g, x, etypes)
        return x



class GraphEnhancer(nn.Module):
    def __init__(self, kge_embedding, input_size, num_rels, gnn_hidden_dim, gnn_num_hidden_layers, adapter_size, output_size=4096, hidden_act='silu'):
        super().__init__()
        self.ent_embeddings = nn.Embedding.from_pretrained(kge_embedding, freeze=True)

        self.rgcn = GNN(
            in_dim=input_size,
            h_dim=gnn_hidden_dim,
            out_dim=input_size,     
            num_rels=num_rels,
            num_hidden_layers=gnn_num_hidden_layers
        )
        
        self.adapter = nn.Sequential(
            nn.Linear(in_features=2*input_size, out_features=adapter_size, bias=False),
            activations.ACT2FN[hidden_act],
            nn.Dropout(p=0.1),
            nn.Linear(in_features=adapter_size, out_features=output_size, bias=False),
        )
        for layer in self.adapter:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, query_ids, entity_ids, subgraph):
        device = query_ids.device
        batch_size = query_ids.size(0)
        K = entity_ids.size(1)

        if (subgraph is None):
            # [original, original]
            flat_qe = torch.cat([query_ids.view(-1,1), entity_ids], dim=1)  
            flat_qe = flat_qe.view(-1)
            base_vec = self.ent_embeddings(flat_qe)
            concat_vec = torch.cat([base_vec, base_vec], dim=-1)
            out_vec = self.adapter(concat_vec)

            out_vec = out_vec.view(batch_size, (K+1), -1) 
            query_embeds = out_vec[:,0,:]
            entity_embeds = out_vec[:,1:,:]
            entity_embeds = entity_embeds.reshape(batch_size*K, -1)
            return query_embeds, entity_embeds

        # Subgraph
        all_query_embeds = []
        all_entity_embeds = []
        for i in range(batch_size):
            edges_list = subgraph[i]  # [[h,r,t], ...]

            if len(edges_list) <= 10:
                # [original, original]
                flat_qe = torch.cat([query_ids[i].view(-1,1), entity_ids[i].unsqueeze(0)], dim=1)
                flat_qe = flat_qe.view(-1) 
                base_vec = self.ent_embeddings(flat_qe)   
                concat_vec = torch.cat([base_vec, base_vec], dim=-1)
                out_vec = self.adapter(concat_vec)        

                q_out = out_vec[0].unsqueeze(0)            
                e_out = out_vec[1:]                        
                all_query_embeds.append(q_out)             
                all_entity_embeds.append(e_out)            
                continue

            q_id = query_ids[i].item()
            ent_list = entity_ids[i]     

            edges_arr = np.array(edges_list)
            src = edges_arr[:,0]
            r   = edges_arr[:,1]
            dst = edges_arr[:,2]

            node_ids_sub = np.unique(np.concatenate([src, dst]))
            node_id_to_idx = {old: idx for idx, old in enumerate(node_ids_sub)}

            mapped_src = [node_id_to_idx[s] for s in src]
            mapped_dst = [node_id_to_idx[d] for d in dst]
            g = dgl.graph((mapped_src, mapped_dst), num_nodes=len(node_ids_sub))
            g = g.to(device)
            etypes = torch.LongTensor(r).to(device)

            node_ids_sub_t = torch.LongTensor(node_ids_sub).to(device)
            base_emb_sub   = self.ent_embeddings(node_ids_sub_t)
            updated_emb_sub = self.rgcn(g, base_emb_sub, etypes) 

            # query
            if q_id in node_id_to_idx:
                # [original, updated]
                q_idx = node_id_to_idx[q_id]
                q_base = base_emb_sub[q_idx].unsqueeze(0)
                q_upd  = updated_emb_sub[q_idx].unsqueeze(0)
            else:
                # [original, original]
                q_base = self.ent_embeddings(torch.LongTensor([q_id]).to(device))
                q_upd  = q_base
            
            q_concat = torch.cat([q_base, q_upd], dim=-1)
            q_out = self.adapter(q_concat) 

            # candiadtes
            e_base_list = []
            e_upd_list  = []
            for e_id in ent_list:
                eid_int = e_id.item()
                if eid_int in node_id_to_idx:
                    # [original, updated]
                    e_idx = node_id_to_idx[eid_int]
                    base_e = base_emb_sub[e_idx]
                    upd_e  = updated_emb_sub[e_idx]
                else:
                    # [original, original]
                    base_e = self.ent_embeddings(e_id.view(-1)).squeeze(0)
                    upd_e  = base_e
                e_base_list.append(base_e.unsqueeze(0))
                e_upd_list.append(upd_e.unsqueeze(0))

            e_base_stack = torch.cat(e_base_list, dim=0) 
            e_upd_stack = torch.cat(e_upd_list, dim=0)  
            e_concat = torch.cat([e_base_stack, e_upd_stack], dim=-1)
            e_out = self.adapter(e_concat)     

            all_query_embeds.append(q_out) 
            all_entity_embeds.append(e_out) 

        query_embeds = torch.cat(all_query_embeds, dim=0)  
        entity_embeds = torch.cat(all_entity_embeds, dim=0)
        
        return query_embeds, entity_embeds
        