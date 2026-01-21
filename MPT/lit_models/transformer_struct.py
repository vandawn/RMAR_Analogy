import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import BaseLitModel
from transformers.optimization import get_linear_schedule_with_warmup
from functools import partial
from .utils import LabelSmoothSoftmaxCEV1
from typing import Callable, Iterable, List
import pdb

def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))

def decode(output_ids, tokenizer):
    return lmap(str.strip, tokenizer.batch_decode(output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True))

class TransformerLitModel(BaseLitModel):
    # def __init__(self, model, args, tokenizer=None, data_config={}):
    #     super().__init__(model, args)
    #     self.save_hyperparameters(args)
    #     if args.label_smoothing != 0.0:
    #         self.loss_fn = LabelSmoothSoftmaxCEV1(lb_smooth=args.label_smoothing)
    #     else:
    #         self.loss_fn = nn.CrossEntropyLoss()

    #     self.best_acc = 0
    #     self.first = True
    #     self.tokenizer = tokenizer
    #     self.__dict__.update(data_config)   # update config

    #     # resize the word embedding layer
    #     self.model.resize_token_embeddings(len(self.tokenizer))
    #     self.alpha = args.alpha
        
    #     # +++ 新增：处理需要追踪的实体 +++
    #     self.entities_to_track = args.entities_to_track
    #     if self.entities_to_track:
    #         self.struct_id2entity = data_config.get('struct_id2entity', {})
    #         self.entity_ids_to_track = {k for k, v in self.struct_id2entity.items() if v in self.entities_to_track}
    
    def __init__(self, model, args, tokenizer=None, data_config={}):
        super().__init__(model, args)
        self.save_hyperparameters(args)
        if args.label_smoothing != 0.0:
            self.loss_fn = LabelSmoothSoftmaxCEV1(lb_smooth=args.label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.best_acc = 0
        self.first = True
        self.tokenizer = tokenizer
        self.__dict__.update(data_config)   # update config

        # resize the word embedding layer
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.alpha = args.alpha

        # ===== ablation flags (int->bool) =====
        self.use_structure   = bool(getattr(args, "use_structure", 1))
        self.use_r_enhance   = bool(getattr(args, "use_r_enhance", 1))
        self.use_memory_bank = bool(getattr(args, "use_memory_bank", 1))

        # tracking entities (optional)
        self.entities_to_track = args.entities_to_track
        if self.entities_to_track:
            self.struct_id2entity = data_config.get('struct_id2entity', {})
            self.entity_ids_to_track = {k for k, v in self.struct_id2entity.items() if v in self.entities_to_track}



    def _init_relation_word(self):
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': ["[R]"]})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.decode = partial(decode, tokenizer=self.tokenizer)
        with torch.no_grad():
            word_embeddings = self.model.get_input_embeddings()
            continous_label_word = self.analogy_relation_ids
            
            rel_word = [a[0] for a in self.tokenizer(["[R]"], add_special_tokens=False)['input_ids']]
            for i, idx in enumerate(rel_word):
                word_embeddings.weight[rel_word[i]] = torch.mean(word_embeddings.weight[continous_label_word], dim=0)
            
            assert torch.equal(self.model.get_input_embeddings().weight, word_embeddings.weight)
            assert torch.equal(self.model.get_input_embeddings().weight, self.model.get_output_embeddings().weight)
            
    # +++ 新增辅助函数 +++
    def _calculate_relation_embeddings(self, hidden_states, h_idx, t_idx):
        """根据头尾实体状态计算关系嵌入 (r_emb = h_state - t_state)"""
        #pdb.set_trace()  # 调试用
        h_states = hidden_states.gather(1, h_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, hidden_states.size(-1))).squeeze(1)
        t_states = hidden_states.gather(1, t_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, hidden_states.size(-1))).squeeze(1)
        return h_states - t_states

    # @torch.no_grad()
    # def _find_and_print_neighbors(self, anchor_r_embs, anchor_rel_labels, batch_idx, batch_struct_q_head_ids, batch_struct_q_tail_ids):
    #     """为当前batch中的每个锚点, 寻找k个近邻, 并按要求打印（支持训练和推理阶段）。"""
    #     # 邻居候选集
    #     candidates_r_embs = torch.cat([anchor_r_embs, self.model.memory_r_emb.clone()], dim=0)
    #     candidates_rel_labels = torch.cat([anchor_rel_labels, self.model.memory_rel_label.clone()], dim=0)

    #     # 规范化后计算余弦相似度
    #     anchor_r_embs_norm = F.normalize(anchor_r_embs, p=2, dim=1)
    #     candidates_r_embs_norm = F.normalize(candidates_r_embs, p=2, dim=1)
    #     sim_matrix = torch.matmul(anchor_r_embs_norm, candidates_r_embs_norm.T)
        
    #     # 找到top-k+1个最相似的（因为自己与自己最相似）
    #     k = self.hparams.num_neighbors
    #     top_k_sims, top_k_indices = torch.topk(sim_matrix, k=k + 1, largest=True, dim=1)
        
    #     # --- 新增的打印判断逻辑 ---
    #     print_condition_met = False
    #     if self.training:
    #         # 训练阶段：按 print_freq 频率打印
    #         if self.global_step % self.hparams.print_freq == 0:
    #             print_condition_met = True
    #     else:
    #         # 推理阶段：只打印每个 epoch 的前2个 batch，避免刷屏
    #         if batch_idx < 2:
    #             print_condition_met = True
        
    #     # 如果满足打印条件
    #     # if print_condition_met:
    #     #     stage = "Training" if self.training else "Inference"
    #     #     print("\n" + "="*20 + f" Step {self.global_step} ({stage}): Neighbor Matching " + "="*20)
            
    #     #     # 只打印当前批次的前5个样本作为示例
    #     #     for i in range(min(anchor_r_embs.size(0), 5)):
    #     #         # --- 新增的名称查找逻辑 ---
    #     #         rel_id = anchor_rel_labels[i].item()
    #     #         relation_name = self.id2relation.get(rel_id, f"ID_{rel_id}")

    #     #         q_head_struct_id = batch_struct_q_head_ids[i].item()
    #     #         q_tail_struct_id = batch_struct_q_tail_ids[i].item()
    #     #         head_name = self.struct_id2entity.get(q_head_struct_id, f"ID_{q_head_struct_id}")
    #     #         tail_name = self.struct_id2entity.get(q_tail_struct_id, f"ID_{q_tail_struct_id}")
                
    #     #         print(f"\n--- Anchor [{i}] | Example: ({head_name}, {relation_name}, {tail_name}) ---")
    #     #         print(f"  Real Relation: {relation_name} (ID: {rel_id})")
    #     #         print(f"  Found {k} Nearest Neighbors (skipping self):")

    #     #         for j in range(1, k + 1):
    #     #             neighbor_global_idx = top_k_indices[i, j].item()
    #     #             neighbor_sim = top_k_sims[i, j].item()
    #     #             neighbor_rel_id = candidates_rel_labels[neighbor_global_idx].item()
    #     #             neighbor_rel_name = self.id2relation.get(neighbor_rel_id, f"ID_{neighbor_rel_id}")
    #     #             source = "Batch" if neighbor_global_idx < anchor_r_embs.size(0) else "Memory"
    #     #             print(f"    - Neighbor {j}: Relation: {neighbor_rel_name} (From: {source}) | Similarity: {neighbor_sim:.4f}")
    #     #     print("="*80 + "\n")

    #     return top_k_indices[:, 1:]
    
    @torch.no_grad()
    def _find_and_print_neighbors(self, anchor_r_embs, anchor_rel_labels, batch_idx, batch_struct_q_head_ids, batch_struct_q_tail_ids):
        """为当前batch中的每个锚点, 寻找k个近邻；支持 w/o Memory Bank。"""
        if self.use_memory_bank:
            candidates_r_embs = torch.cat([anchor_r_embs, self.model.memory_r_emb.clone()], dim=0)
            candidates_rel_labels = torch.cat([anchor_rel_labels, self.model.memory_rel_label.clone()], dim=0)
        else:
            # 不使用记忆库，仅用本 batch
            candidates_r_embs = anchor_r_embs
            candidates_rel_labels = anchor_rel_labels

        # cosine sim
        anchor_r_embs_norm = F.normalize(anchor_r_embs, p=2, dim=1)
        candidates_r_embs_norm = F.normalize(candidates_r_embs, p=2, dim=1)
        sim_matrix = torch.matmul(anchor_r_embs_norm, candidates_r_embs_norm.T)

        k = self.hparams.num_neighbors
        top_k = min(k + 1, candidates_r_embs.size(0))  # +1 用于跳过自身
        top_k_sims, top_k_indices = torch.topk(sim_matrix, k=top_k, largest=True, dim=1)

        # 返回去除“自身”的邻居索引
        return top_k_indices[:, 1:]


    def forward(self, x):
        return self.model(x)
    
    def _enhance_relation_representation(self, trans_hidden_states, q_head_idx, q_tail_idx, rel_idx, rel_label, batch_idx, bsz, batch_struct_q_head_ids, batch_struct_q_tail_ids):
        """
        根据给定的隐藏状态，动态查找邻居并使用注意力融合来增强关系表示。
        此方法在训练和推理阶段均可复用。
        """
        # 1. 计算当前批次的关系嵌入 r_emb (h_state - t_state)
        r_emb_online = self._calculate_relation_embeddings(trans_hidden_states, q_head_idx, q_tail_idx)

        # 2. 动态查找k个近邻
        # ** FIX IS HERE: Pass the new entity IDs to the print function **
        neighbor_indices = self._find_and_print_neighbors(r_emb_online, rel_label, batch_idx, batch_struct_q_head_ids, batch_struct_q_tail_ids)

        # 3. 准备锚点和邻居的 [R] 表示用于注意力融合
        # R_fine 是从 transform head 输出的细粒度 [R] 表示
        R_fine_online = trans_hidden_states[torch.arange(bsz), rel_idx[:, 0]]
        
        # 邻居的 [R] 表示从记忆库中获取
        # 注意：这里我们将当前批次的 R_fine_online 也作为邻居候选，以处理邻居在当前批次内的情况
        candidates_R_fine = torch.cat([R_fine_online, self.model.memory_R_fine.clone()], dim=0)
        neighbor_R_fines = candidates_R_fine[neighbor_indices]
        
        # 4. 使用注意力机制进行融合
        anchor_R_fine_for_attn = R_fine_online.unsqueeze(1)
        # R_coarse 是通过注意力加权求和得到的粗粒度补充信息
        R_coarse, _ = self.model.neighborhood_attention(query=anchor_R_fine_for_attn, key=neighbor_R_fines, value=neighbor_R_fines)
        
        # 5. 通过残差连接得到最终的增强关系表示
        R_enhanced_batch = R_fine_online + R_coarse.squeeze(1)

        # 6. 获取查询对的 [R] 表示
        query_R_fine = trans_hidden_states[torch.arange(bsz), rel_idx[:, 1]]
        
        return R_enhanced_batch, query_R_fine

    
    # def training_step(self, batch, batch_idx):
    #     # +++ 在调用模型前，弹出所有 LitModel 专用字段 +++
    #     label = batch.pop("label")
    #     rel_label = batch.pop('rel_label', None)
    #     pre_type = batch.pop('pre_type', None)
    #     rel_idx = batch.pop('rel_idx', None)
    #     q_head_idx = batch.pop('q_head_idx', None)
    #     a_head_idx = batch.pop('a_head_idx', None)
    #     # need pop
    #     q_tail_idx = batch.pop('q_tail_idx', None)
    #     # +++++++++++++++++++++++++++++++++++++++++++++++

    #     input_ids = batch['input_ids']
    #     # 现在 batch 中只剩下模型 forward 函数认识的参数了
    #     model_output_tuple = self.model(**batch, return_dict=True)
    #     model_output = model_output_tuple[0]
    #     logits = model_output.logits
    #     bs = input_ids.shape[0]

    #     if self.args.pretrain:
    #         _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
    #         assert mask_idx.shape[0] == bs, "only one mask in sequence!"
    #         mask_logits = logits[torch.arange(bs), mask_idx]

    #         entity_loss, relation_loss = 0, 0
    #         entity_mask = (pre_type != 2).nonzero(as_tuple=True)[0]
    #         if len(entity_mask) > 0:
    #             entity_logits = mask_logits[entity_mask, self.entity_id_st:self.entity_id_ed]
    #             entity_label = label[entity_mask]
    #             entity_loss = self.loss_fn(entity_logits, entity_label)

    #         relation_mask = (pre_type == 2).nonzero(as_tuple=True)[0]
    #         if len(relation_mask) > 0:
    #             relation_logits = mask_logits[relation_mask, self.relation_id_st:self.relation_id_ed]
    #             relation_label = label[relation_mask]
    #             relation_loss = self.loss_fn(relation_logits, relation_label)
            
    #         loss = entity_loss + relation_loss
            
    #     else:
    #         # 1. 主模型(Online)前向传播
    #         outputs_online, trans_hidden_states_online, _ = self.model(**batch, return_dict=True)
    #         logits = outputs_online.logits
            
    #         # 2. 动量模型(Momentum)前向传播 (仅在训练时需要)
    #         with torch.no_grad():
    #             self.model._update_momentum_encoder()
    #             # 步骤 1: 调用基础的动量编码器 unimo_m，它返回2个值
    #             outputs_momentum, _ = self.model.unimo_m(**batch, return_dict=True)
    #             sequence_output_m = outputs_momentum[0] # 或者 outputs_momentum.last_hidden_state
                
    #             # 步骤 2: 将 unimo_m 的输出传入分类头(cls)，以生成我们需要的 trans_hidden_states_m
    #             # 注意：这里我们复用 online 模型的 self.model.cls 即可，因为它在 no_grad 上下文中只是做一次前向计算
    #             _, trans_hidden_states_m = self.model.cls(sequence_output_m)

    #         # 3. 调用新方法获取增强的关系表示
    #         R_enhanced_batch, query_R_fine = self._enhance_relation_representation(
    #             trans_hidden_states=trans_hidden_states_online,
    #             q_head_idx=q_head_idx,
    #             q_tail_idx=q_tail_idx,
    #             rel_idx=rel_idx,
    #             rel_label=rel_label,
    #             batch_idx=batch_idx,
    #             bsz=bs,
    #             batch_struct_q_head_ids=batch['struct_head_id'], # 传入实体ID
    #             batch_struct_q_tail_ids=batch['struct_tail_id']  # 传入实体ID
    #         )

    #         # 4. 计算总损失
    #         _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
    #         mask_logits = logits[torch.arange(bs), mask_idx]
    #         scaled_mask_logits = mask_logits / self.hparams.temperature
    #         loss_link_pred = self.loss_fn(scaled_mask_logits[:, self.analogy_entity_ids], label)
            
    #         sim_loss = (1 - F.cosine_similarity(R_enhanced_batch, query_R_fine)).mean()
            
    #         loss = loss_link_pred + self.hparams.alpha * sim_loss

    #         # 5. 更新记忆库 (仅在训练时需要)
    #         with torch.no_grad():
    #             r_emb_momentum = self._calculate_relation_embeddings(trans_hidden_states_m, q_head_idx, q_tail_idx)
                
    #             R_fine_momentum = trans_hidden_states_m[torch.arange(bs), rel_idx[:, 0]]
    #             self.model._update_memory_bank(r_emb_momentum, R_fine_momentum, rel_label)

    #     if batch_idx == 0:
    #         print('\n'.join(self.decode(batch['input_ids'][:4])))
    #     return loss
    
    def training_step(self, batch, batch_idx):
        # 弹出 lit 专用字段
        label = batch.pop("label")
        rel_label = batch.pop('rel_label', None)
        pre_type = batch.pop('pre_type', None)
        rel_idx = batch.pop('rel_idx', None)
        q_head_idx = batch.pop('q_head_idx', None)
        a_head_idx = batch.pop('a_head_idx', None)
        q_tail_idx = batch.pop('q_tail_idx', None)

        input_ids = batch['input_ids']
        model_output_tuple = self.model(**batch, return_dict=True)
        model_output = model_output_tuple[0]
        logits = model_output.logits
        bs = input_ids.shape[0]

        if self.args.pretrain:
            _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
            assert mask_idx.shape[0] == bs, "only one mask in sequence!"
            mask_logits = logits[torch.arange(bs), mask_idx]

            entity_loss, relation_loss = 0, 0
            entity_mask = (pre_type != 2).nonzero(as_tuple=True)[0]
            if len(entity_mask) > 0:
                entity_logits = mask_logits[entity_mask, self.entity_id_st:self.entity_id_ed]
                entity_label = label[entity_mask]
                entity_loss = self.loss_fn(entity_logits, entity_label)

            relation_mask = (pre_type == 2).nonzero(as_tuple=True)[0]
            if len(relation_mask) > 0:
                relation_logits = mask_logits[relation_mask, self.relation_id_st:self.relation_id_ed]
                relation_label = label[relation_mask]
                relation_loss = self.loss_fn(relation_logits, relation_label)

            loss = entity_loss + relation_loss

        else:
            # 1) online forward
            outputs_online, trans_hidden_states_online, _ = self.model(**batch, return_dict=True)
            logits = outputs_online.logits

            # 2) （可选）邻域增强 & 动量
            if self.use_r_enhance:
                with torch.no_grad():
                    self.model._update_momentum_encoder()
                    outputs_momentum, _ = self.model.unimo_m(**batch, return_dict=True)
                    sequence_output_m = outputs_momentum[0]
                    _, trans_hidden_states_m = self.model.cls(sequence_output_m)

                R_enhanced_batch, query_R_fine = self._enhance_relation_representation(
                    trans_hidden_states=trans_hidden_states_online,
                    q_head_idx=q_head_idx,
                    q_tail_idx=q_tail_idx,
                    rel_idx=rel_idx,
                    rel_label=rel_label,
                    batch_idx=batch_idx,
                    bsz=bs,
                    batch_struct_q_head_ids=batch['struct_head_id'],
                    batch_struct_q_tail_ids=batch['struct_tail_id']
                )
            else:
                R_enhanced_batch = None
                query_R_fine = None

            # 3) 链接预测损失
            _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
            mask_logits = logits[torch.arange(bs), mask_idx]
            logits_subset = (mask_logits[:, self.analogy_entity_ids]) / self.hparams.temperature
            loss_link_pred = self.loss_fn(logits_subset, label)

            if self.use_r_enhance:
                sim_loss = (1 - F.cosine_similarity(R_enhanced_batch, query_R_fine)).mean()
                loss = loss_link_pred + self.hparams.alpha * sim_loss
            else:
                loss = loss_link_pred

            # 4) （可选）更新记忆库
            if self.use_r_enhance and self.use_memory_bank:
                with torch.no_grad():
                    r_emb_momentum = self._calculate_relation_embeddings(trans_hidden_states_m, q_head_idx, q_tail_idx)
                    R_fine_momentum = trans_hidden_states_m[torch.arange(bs), rel_idx[:, 0]]
                    self.model._update_memory_bank(r_emb_momentum, R_fine_momentum, rel_label)

        if batch_idx == 0:
            print('\n'.join(self.decode(batch['input_ids'][:4])))
        return loss

    
    


    
    
    # def validation_step(self, batch, batch_idx):
    #     # 弹出所有 LitModel 专用字段，但保存它们以供后续使用
    #     label = batch.pop("label")
    #     rel_label = batch.pop('rel_label', None)
    #     pre_type = batch.pop('pre_type', None)
    #     rel_idx = batch.pop('rel_idx', None)
    #     q_head_idx = batch.pop('q_head_idx', None)
    #     a_head_idx = batch.pop('a_head_idx', None)
    #     q_tail_idx = batch.pop('q_tail_idx', None)

    #     input_ids = batch['input_ids']
    #     bsz = input_ids.shape[0]
        
    #     # 1. 调用模型获取中间表示
    #     # model.forward() 返回 (MaskedLMOutput, trans_hidden_states, gates)
    #     model_outputs, trans_hidden_states, gates = self.model(**batch, return_dict=True)

    #     if self.args.pretrain:
    #         # 预训练阶段的评估逻辑不变
    #         logits = model_outputs.logits
    #         _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
    #         mask_logits = logits[torch.arange(bsz), mask_idx]
        
    #         entity_ranks, relation_ranks = None, None
    #         entity_mask = (pre_type != 2).nonzero(as_tuple=True)[0]
    #         if len(entity_mask) > 0:
    #             entity_logits = mask_logits[entity_mask, self.entity_id_st:self.entity_id_ed]
    #             entity_label = label[entity_mask]
    #             _, entity_outputs = torch.sort(entity_logits, dim=1, descending=True)
    #             _, entity_outputs = torch.sort(entity_outputs, dim=1)
    #             entity_ranks = entity_outputs[torch.arange(entity_mask.size(0)), entity_label].detach().cpu() + 1
    #         relation_mask = (pre_type == 2).nonzero(as_tuple=True)[0]
    #         if len(relation_mask) > 0:
    #             relation_logits = mask_logits[relation_mask, self.relation_id_st:self.relation_id_ed]
    #             relation_label = label[relation_mask]
    #             _, relation_outputs = torch.sort(relation_logits, dim=1, descending=True)
    #             _, relation_outputs = torch.sort(relation_outputs, dim=1)
    #             relation_ranks = relation_outputs[torch.arange(relation_mask.size(0)), relation_label].detach().cpu() + 1
            
    #         result = {}
    #         if entity_ranks is not None: result['entity_ranks'] = np.array(entity_ranks)
    #         if relation_ranks is not None: result['relation_ranks'] = np.array(relation_ranks)
    #     else:
    #         # 2. 调用邻域增强方法
    #         R_enhanced_batch, _ = self._enhance_relation_representation(
    #             trans_hidden_states=trans_hidden_states,
    #             q_head_idx=q_head_idx,
    #             q_tail_idx=q_tail_idx,
    #             rel_idx=rel_idx,
    #             rel_label=rel_label,
    #             batch_idx=batch_idx,
    #             bsz=bsz,
    #             batch_struct_q_head_ids=batch['struct_head_id'], # 传入实体ID
    #             batch_struct_q_tail_ids=batch['struct_tail_id']  # 传入实体ID
    #         )
            
    #         # 3. 将增强后的表示注入，并重新计算预测 logits
    #         # 创建 trans_hidden_states 的副本以进行修改
    #         trans_hidden_states_enhanced = trans_hidden_states.clone()
            
    #         # 获取查询对中 [R] token 的位置
    #         query_r_indices = rel_idx[:, 1] 
    #         # 注入操作
    #         for i in range(bsz):
    #             trans_hidden_states_enhanced[i, query_r_indices[i], :] = R_enhanced_batch[i]
            
    #         # 使用增强后的隐藏状态通过模型的最后解码层得到新的 logits
    #         # UnimoLMPredictionHead.decoder
    #         logits = self.model.cls.predictions.decoder(trans_hidden_states_enhanced)

    #         # 4. 使用新的 logits 进行评估
    #         _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
    #         mask_logits = logits[torch.arange(bsz), mask_idx][:, self.analogy_entity_ids]
    #         scaled_mask_logits = mask_logits / self.hparams.temperature
    #         _, outputs1 = torch.sort(scaled_mask_logits, dim=1, descending=True)
    #         _, outputs = torch.sort(outputs1, dim=1)
    #         entity_ranks = outputs[torch.arange(bsz), label].detach().cpu() + 1
    #         result = dict(entity_ranks=np.array(entity_ranks))
    #     # --- 评估逻辑结束 ---

    #     # --- 日志记录逻辑 ---
    #     if self.args.fusion_strategy == 'gate' and gates:
    #         avg_gates = {name: g.mean() for name, g in gates.items()}
    #         self.log_dict({f"avg_gate/{name}": val for name, val in avg_gates.items()}, on_step=False, on_epoch=True)

    #         if self.entities_to_track:
    #             struct_head_ids = batch.get('struct_head_id')
    #             text_head_gates = gates.get('text_head_gate')

    #             if struct_head_ids is not None and text_head_gates is not None:
    #                 for i in range(struct_head_ids.size(0)):
    #                     entity_id = struct_head_ids[i].item()
    #                     if entity_id in self.entity_ids_to_track:
    #                         entity_name = self.struct_id2entity[entity_id]
    #                         gate_value = text_head_gates[i].mean().item()
    #                         self.log(f"entity_gate/{entity_name}", gate_value, on_step=False, on_epoch=True)
    #     # --- 日志记录结束 ---
        
    #     return result
    
    def validation_step(self, batch, batch_idx):
        # 弹出 lit 专用字段
        label = batch.pop("label")
        rel_label = batch.pop('rel_label', None)
        pre_type = batch.pop('pre_type', None)
        rel_idx = batch.pop('rel_idx', None)
        q_head_idx = batch.pop('q_head_idx', None)
        a_head_idx = batch.pop('a_head_idx', None)
        q_tail_idx = batch.pop('q_tail_idx', None)

        input_ids = batch['input_ids']
        bsz = input_ids.shape[0]

        # 1) 获取中间表示
        model_outputs, trans_hidden_states, gates = self.model(**batch, return_dict=True)

        if self.args.pretrain:
            logits = model_outputs.logits
            _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
            mask_logits = logits[torch.arange(bsz), mask_idx]

            entity_ranks, relation_ranks = None, None
            entity_mask = (pre_type != 2).nonzero(as_tuple=True)[0]
            if len(entity_mask) > 0:
                entity_logits = mask_logits[entity_mask, self.entity_id_st:self.entity_id_ed]
                entity_label = label[entity_mask]
                _, entity_outputs = torch.sort(entity_logits, dim=1, descending=True)
                _, entity_outputs = torch.sort(entity_outputs, dim=1)
                entity_ranks = entity_outputs[torch.arange(entity_mask.size(0)), entity_label].detach().cpu() + 1

            relation_mask = (pre_type == 2).nonzero(as_tuple=True)[0]
            if len(relation_mask) > 0:
                relation_logits = mask_logits[relation_mask, self.relation_id_st:self.relation_id_ed]
                relation_label = label[relation_mask]
                _, relation_outputs = torch.sort(relation_logits, dim=1, descending=True)
                _, relation_outputs = torch.sort(relation_outputs, dim=1)
                relation_ranks = relation_outputs[torch.arange(relation_mask.size(0)), relation_label].detach().cpu() + 1

            result = {}
            if entity_ranks is not None: result['entity_ranks'] = np.array(entity_ranks)
            if relation_ranks is not None: result['relation_ranks'] = np.array(relation_ranks)

        else:
            # 2) （可选）R 增强 -> 注入 -> 解码
            if self.use_r_enhance:
                R_enhanced_batch, _ = self._enhance_relation_representation(
                    trans_hidden_states=trans_hidden_states,
                    q_head_idx=q_head_idx,
                    q_tail_idx=q_tail_idx,
                    rel_idx=rel_idx,
                    rel_label=rel_label,
                    batch_idx=batch_idx,
                    bsz=bsz,
                    batch_struct_q_head_ids=batch['struct_head_id'],
                    batch_struct_q_tail_ids=batch['struct_tail_id']
                )
                trans_hidden_states_enhanced = trans_hidden_states.clone()
                query_r_indices = rel_idx[:, 1]
                trans_hidden_states_enhanced[torch.arange(bsz), query_r_indices] = R_enhanced_batch
                logits = self.model.cls.predictions.decoder(trans_hidden_states_enhanced)
            else:
                logits = self.model.cls.predictions.decoder(trans_hidden_states)

            # 3) 评估
            _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
            mask_logits = logits[torch.arange(bsz), mask_idx][:, self.analogy_entity_ids]
            mask_logits = mask_logits / self.hparams.temperature
            _, outputs1 = torch.sort(mask_logits, dim=1, descending=True)
            _, outputs = torch.sort(outputs1, dim=1)
            entity_ranks = outputs[torch.arange(bsz), label].detach().cpu() + 1
            result = dict(entity_ranks=np.array(entity_ranks))

        # 日志（门控）
        if self.args.fusion_strategy == 'gate' and gates:
            avg_gates = {name: g.mean() for name, g in gates.items()}
            self.log_dict({f"avg_gate/{name}": val for name, val in avg_gates.items()}, on_step=False, on_epoch=True)

            if self.entities_to_track:
                struct_head_ids = batch.get('struct_head_id')
                text_head_gates = gates.get('text_head_gate')
                if struct_head_ids is not None and text_head_gates is not None:
                    for i in range(struct_head_ids.size(0)):
                        entity_id = struct_head_ids[i].item()
                        if entity_id in self.entity_ids_to_track:
                            entity_name = self.struct_id2entity[entity_id]
                            gate_value = text_head_gates[i].mean().item()
                            self.log(f"entity_gate/{entity_name}", gate_value, on_step=False, on_epoch=True)

        return result

    
    

    def validation_epoch_end(self, outputs) -> None:
        entity_ranks = [_['entity_ranks'] for _ in outputs if 'entity_ranks' in _]
        if len(entity_ranks) > 0:
            entity_ranks = np.concatenate(entity_ranks)

            # entity
            hits20 = (entity_ranks<=20).mean()
            hits10 = (entity_ranks<=10).mean()
            hits5 = (entity_ranks<=5).mean()
            hits3 = (entity_ranks<=3).mean()
            hits1 = (entity_ranks<=1).mean()

            self.log("Eval_entity/hits1", hits1)
            self.log("Eval_entity/hits3", hits3)
            self.log("Eval_entity/hits5", hits5)
            self.log("Eval_entity/hits10", hits10)
            self.log("Eval_entity/hits20", hits20)
            self.log("Eval_entity/mean_rank", entity_ranks.mean())
            self.log("Eval_entity/mrr", (1. / entity_ranks).mean())
            self.log("entity_hits10", hits10, prog_bar=True)
            self.log("entity_hits1", hits1, prog_bar=True)
            
    

    
    # def test_step(self, batch, batch_idx):
    #     # 弹出所有 LitModel 专用字段
    #     label = batch.pop("label")
    #     rel_label = batch.pop('rel_label', None)
    #     pre_type = batch.pop('pre_type', None)
    #     rel_idx = batch.pop('rel_idx', None)
    #     q_head_idx = batch.pop('q_head_idx', None)
    #     a_head_idx = batch.pop('a_head_idx', None)
    #     q_tail_idx = batch.pop('q_tail_idx', None)

    #     input_ids = batch['input_ids']
    #     bsz = input_ids.shape[0]

    #     # 1. 调用模型获取中间表示
    #     # ** FIX IS HERE: The output is correctly named 'model_outputs' (plural) **
    #     model_outputs, trans_hidden_states, _ = self.model(**batch, return_dict=True)

    #     if self.args.pretrain:
    #         # 预训练阶段的评估逻辑
    #         # ** The variable 'model_outputs' (plural) is used correctly here **
    #         logits = model_outputs.logits
    #         _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
    #         mask_logits = logits[torch.arange(bsz), mask_idx]
        
    #         entity_ranks, relation_ranks = None, None
    #         entity_mask = (pre_type != 2).nonzero(as_tuple=True)[0]
    #         if len(entity_mask) > 0:
    #             entity_logits = mask_logits[entity_mask, self.entity_id_st:self.entity_id_ed]
    #             entity_label = label[entity_mask]
    #             _, entity_outputs = torch.sort(entity_logits, dim=1, descending=True)
    #             _, entity_outputs = torch.sort(entity_outputs, dim=1)
    #             entity_ranks = entity_outputs[torch.arange(entity_mask.size(0)), entity_label].detach().cpu() + 1
            
    #         relation_mask = (pre_type == 2).nonzero(as_tuple=True)[0]
    #         if len(relation_mask) > 0:
    #             relation_logits = mask_logits[relation_mask, self.relation_id_st:self.relation_id_ed]
    #             relation_label = label[relation_mask]
    #             _, relation_outputs = torch.sort(relation_logits, dim=1, descending=True)
    #             _, relation_outputs = torch.sort(relation_outputs, dim=1)
    #             relation_ranks = relation_outputs[torch.arange(relation_mask.size(0)), relation_label].detach().cpu() + 1
            
    #         result = {}
    #         if entity_ranks is not None:
    #             result['entity_ranks'] = np.array(entity_ranks)
    #         if relation_ranks is not None:
    #             result['relation_ranks'] = np.array(relation_ranks)
    #     else: # Fine-tuning
    #         # 2. 调用邻域增强方法
    #         R_enhanced_batch, _ = self._enhance_relation_representation(
    #             trans_hidden_states=trans_hidden_states,
    #             q_head_idx=q_head_idx,
    #             q_tail_idx=q_tail_idx,
    #             rel_idx=rel_idx,
    #             rel_label=rel_label,
    #             batch_idx=batch_idx,
    #             bsz=bsz,
    #             batch_struct_q_head_ids=batch['struct_head_id'],
    #             batch_struct_q_tail_ids=batch['struct_tail_id']
    #         )
            
    #         # 3. 注入增强表示并重新计算 logits
    #         trans_hidden_states_enhanced = trans_hidden_states.clone()
    #         query_r_indices = rel_idx[:, 1]
    #         for i in range(bsz):
    #             trans_hidden_states_enhanced[i, query_r_indices[i], :] = R_enhanced_batch[i]
            
    #         logits = self.model.cls.predictions.decoder(trans_hidden_states_enhanced)

    #         # 4. 使用新 logits 进行评估
    #         _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
    #         mask_logits = logits[torch.arange(bsz), mask_idx][:, self.analogy_entity_ids]
    #         scaled_mask_logits = mask_logits / self.hparams.temperature
    #         _, outputs1 = torch.sort(scaled_mask_logits, dim=1, descending=True)
    #         _, outputs = torch.sort(outputs1, dim=1)
    #         entity_ranks = outputs[torch.arange(bsz), label].detach().cpu() + 1
    #         result = dict(entity_ranks=np.array(entity_ranks))
        
    #     return result
    
    def test_step(self, batch, batch_idx):
        # 弹出 lit 专用字段
        label = batch.pop("label")
        rel_label = batch.pop('rel_label', None)
        pre_type = batch.pop('pre_type', None)
        rel_idx = batch.pop('rel_idx', None)
        q_head_idx = batch.pop('q_head_idx', None)
        a_head_idx = batch.pop('a_head_idx', None)
        q_tail_idx = batch.pop('q_tail_idx', None)

        input_ids = batch['input_ids']
        bsz = input_ids.shape[0]

        # 1) 获取中间表示
        model_outputs, trans_hidden_states, _ = self.model(**batch, return_dict=True)

        if self.args.pretrain:
            logits = model_outputs.logits
            _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
            mask_logits = logits[torch.arange(bsz), mask_idx]

            entity_ranks, relation_ranks = None, None
            entity_mask = (pre_type != 2).nonzero(as_tuple=True)[0]
            if len(entity_mask) > 0:
                entity_logits = mask_logits[entity_mask, self.entity_id_st:self.entity_id_ed]
                entity_label = label[entity_mask]
                _, entity_outputs = torch.sort(entity_logits, dim=1, descending=True)
                _, entity_outputs = torch.sort(entity_outputs, dim=1)
                entity_ranks = entity_outputs[torch.arange(entity_mask.size(0)), entity_label].detach().cpu() + 1

            relation_mask = (pre_type == 2).nonzero(as_tuple=True)[0]
            if len(relation_mask) > 0:
                relation_logits = mask_logits[relation_mask, self.relation_id_st:self.relation_id_ed]
                relation_label = label[relation_mask]
                _, relation_outputs = torch.sort(relation_logits, dim=1, descending=True)
                _, relation_outputs = torch.sort(relation_outputs, dim=1)
                relation_ranks = relation_outputs[torch.arange(relation_mask.size(0)), relation_label].detach().cpu() + 1

            result = {}
            if entity_ranks is not None:
                result['entity_ranks'] = np.array(entity_ranks)
            if relation_ranks is not None:
                result['relation_ranks'] = np.array(relation_ranks)
        else:
            # 2) （可选）R 增强
            if self.use_r_enhance:
                R_enhanced_batch, _ = self._enhance_relation_representation(
                    trans_hidden_states=trans_hidden_states,
                    q_head_idx=q_head_idx,
                    q_tail_idx=q_tail_idx,
                    rel_idx=rel_idx,
                    rel_label=rel_label,
                    batch_idx=batch_idx,
                    bsz=bsz,
                    batch_struct_q_head_ids=batch['struct_head_id'],
                    batch_struct_q_tail_ids=batch['struct_tail_id']
                )
                trans_hidden_states_enhanced = trans_hidden_states.clone()
                query_r_indices = rel_idx[:, 1]
                trans_hidden_states_enhanced[torch.arange(bsz), query_r_indices] = R_enhanced_batch
                logits = self.model.cls.predictions.decoder(trans_hidden_states_enhanced)
            else:
                logits = self.model.cls.predictions.decoder(trans_hidden_states)

            # 3) 评估
            _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
            mask_logits = logits[torch.arange(bsz), mask_idx][:, self.analogy_entity_ids]
            mask_logits = mask_logits / self.hparams.temperature
            _, outputs1 = torch.sort(mask_logits, dim=1, descending=True)
            _, outputs = torch.sort(outputs1, dim=1)
            entity_ranks = outputs[torch.arange(bsz), label].detach().cpu() + 1
            result = dict(entity_ranks=np.array(entity_ranks))

        return result


    def test_epoch_end(self, outputs) -> None:
        entity_ranks = [_['entity_ranks'] for _ in outputs if 'entity_ranks' in _]

        if len(entity_ranks) > 0:
            entity_ranks = np.concatenate(entity_ranks)

            # entity
            hits20 = (entity_ranks<=20).mean()
            hits10 = (entity_ranks<=10).mean()
            hits5 = (entity_ranks<=5).mean()
            hits3 = (entity_ranks<=3).mean()
            hits1 = (entity_ranks<=1).mean()

            self.log("Eval_entity/hits1", hits1)
            self.log("Eval_entity/hits3", hits3)
            self.log("Eval_entity/hits5", hits5)
            self.log("Eval_entity/hits10", hits10)
            self.log("Eval_entity/hits20", hits20)
            self.log("Eval_entity/mean_rank", entity_ranks.mean())
            self.log("Eval_entity/mrr", (1. / entity_ranks).mean())
            self.log("entity_hits10", hits10, prog_bar=True)
            self.log("entity_hits1", hits1, prog_bar=True)

    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]

        optimizer_group_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]

        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * self.args.warm_up_radio, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }
    
    def _freeze_attention(self):
        for k, v in self.model.named_parameters():
            if "word" not in k:
                v.requires_grad = False
            else:
                print(k)
    
    def _freeze_word_embedding(self):
        for k, v in self.model.named_parameters():
            if "word" in k:
                print(k)
                v.requires_grad = False

    @staticmethod
    def add_to_argparse(parser):
        parser = BaseLitModel.add_to_argparse(parser)

        parser.add_argument("--label_smoothing", type=float, default=0.1, help="")
        parser.add_argument("--bce", type=int, default=0, help="")
        return parser
