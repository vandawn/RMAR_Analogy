from email.generator import Generator
import torch
import mmkgc
import numpy as np
from torch.utils.data import DataLoader
from torch.utils import data
#from mmkgc.config import Trainer, Tester
from mmkgc.config import Tester
from mmkgc.config.Trainer_gaussian_ft import Trainer
from mmkgc.module.model.AdvMixRotatE_gaussian_ft import AdvMixRotatE

from mmkgc.module.loss import MarginLoss, SigmoidLoss
from mmkgc.module.strategy.NegativeSampling_gaussian_ft import NegativeSampling
from mmkgc.data import TrainDataLoader, TestDataLoader
from torchlight import initialize_exp, get_dump_path
from args import get_args
import os.path as osp
import os
import pdb




class AnalogyFinetuneDataset(data.Dataset):
    """Dataset implementation for handling FB15K and FB15K-237."""

    def __init__(self, data_path: str, entity2id, relation2id):
        self.entity2id = entity2id
        self.relation2id = relation2id
        with open(data_path, "r") as f:
            # data in tuples (head, relation, tail)
            self.data = [line[:-1].split(" ") for line in f]

    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.data)

    def __getitem__(self, index):
        """Returns (head id, relation id, tail id)."""
        eh, et, eq, ea, r, m = self.data[index]

        return int(eh), int(et), int(eq), int(ea), int(r), int(m)
    
def run_analogical_reasoning(self, type_constrain=False):
        ranks = []
        outputs = []
        for data in self.data_loader:
            predictions = self.test_one_step_ft(data)   # bsz, 11292
            truth = data[3]
            _, output = torch.sort(predictions, dim=1, descending=True)
            _, predictions = torch.sort(output, dim=1)
            rank = predictions[torch.arange(truth.shape[0]), truth].detach().cpu() + 1
            ranks.append(rank)
            outputs.append(np.array(output.detach().cpu()))
            
        ranks = torch.cat(ranks).float()
        outputs = np.concatenate(outputs)
        mean_ranks = torch.mean(ranks).item()
        mean_reciprocal_ranks = torch.mean(1. / ranks).item()
        hits_ats = torch.FloatTensor((list(map(
            lambda x: torch.mean((ranks <= x).float()).item(),
            (1, 3, 5, 10)
        ))))
        return mean_reciprocal_ranks, mean_ranks, hits_ats[-1], hits_ats[-2], hits_ats[-3], hits_ats[-4]


def create_mappings(dataset_path: str):
    """Creates separate mappings to indices for entities and relations."""
    with open(f'{dataset_path}/entity2id.txt', 'r') as f:
        lines = f.readlines()
        entity2id = {line.split('\t')[0]:line.split('\t')[1] for line in lines[1:]}
    with open(f'{dataset_path}/relation2id.txt', 'r') as f:
        lines = f.readlines()
        relation2id = {line.split('\t')[0]:line.split('\t')[1] for line in lines[1:]}
    return entity2id, relation2id

if __name__ == "__main__":
    args = get_args()
    this_dir = osp.dirname(__file__)
    # change
    #pdb.set_trace()
    data_root = osp.abspath(osp.join(this_dir, 'data', ''))
    data_path = osp.join(data_root, args.data_path)
    args.dump_path = osp.join(data_path, args.dump_path)
    save_path = osp.join(data_path, "checkpoint")
    
    
    code_gen_file_path = osp.abspath(osp.join(this_dir, 'code_generate_file', ''))
    
    
    

    file_path = os.path.join(os.path.dirname(data_root), 'benchmarks', 'Analogy')
    
    #pdb.set_trace()
    if not osp.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    if not osp.exists(code_gen_file_path):
        os.makedirs(code_gen_file_path, exist_ok=True)
        
    

    args.save = osp.join(save_path, args.save)
    args.exp_name = f"{args.exp_name}-{args.dataset}"

    logger = initialize_exp(args)

    # set the seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    entity2id, relation2id = create_mappings(dataset_path=file_path)
    img_emb = torch.load('./embeddings/' + args.dataset + '-visual.pth')
    text_emb = torch.load('./embeddings/' + args.dataset + '-textual.pth')


    # 生成唯一实验 ID（基于实验超参数）
    exp_id = f"joint{args.joint_way}_noise{args.add_noise}_emb{args.dim}_neg{args.neg_num}_bs{args.batch_size}_margin{args.margin}_layer{args.num_hidden_layers}_head{args.num_attention_heads}"

    # 确保 checkpoint 目录存在
    ckpt_dir_pt = "ckpt/analogy/pretrain"
    ckpt_dir_ft = "ckpt/analogy/finetune"
    os.makedirs(ckpt_dir_pt, exist_ok=True)
    os.makedirs(ckpt_dir_ft, exist_ok=True)

    # 生成 checkpoint 文件路径
    pt_checkpoint_dir = osp.join(ckpt_dir_pt, f"pt_Analogy_ANAG_{exp_id}_T05_wigrad_dual_noise.ckpt")
    ft_checkpoint_dir = osp.join(ckpt_dir_ft, f"ft_Analogy_ANAG_{exp_id}_T05_wigrad_dual_noise.ckpt")

    print(f"Checkpoint will be saved as:\n  Pretrain: {pt_checkpoint_dir}\n  Finetune: {ft_checkpoint_dir}")
    
    pt_exp_id = "pt_Analogy_ANAG_" + exp_id + "_wigrad_dual_noise"
    ft_exp_id = "ft_Analogy_ANAG_" + exp_id + "_wigrad_dual_noise"
    
    # pt_checkpoint_dir = "ckpt/analogy/" +pt_exp_id + ".ckpt"
    # ft_checkpoint_dir = "ckpt/analogy/" +ft_exp_id + ".ckpt"
    
    

    if not args.finetune:
        # dataloader for training
        #pdb.set_trace()
        train_dataloader = TrainDataLoader(
            in_path="./benchmarks/" + args.dataset + '/',
            batch_size=args.batch_size,
            threads=8,
            sampling_mode="normal",
            bern_flag=1,
            filter_flag=1,
            neg_ent=args.neg_num,
            neg_rel=0
        )
        
        for batch in train_dataloader:
            print(batch)
            break  # 只查看第一个批次就足够了，如果你想查看更多批次，可以去掉这行代码
        #pdb.set_trace()
        # dataloader for test
        test_dataloader = TestDataLoader(
            "./benchmarks/" + args.dataset + '/', "link")
        
        fit_head_parameters_tensor = []
        fit_tail_parameters_tensor = []
        total_bsz =[]

        # 为每个relation ID从1到192设置初始参数为空
        for relation_id in range(len(relation2id)):
            fit_head_parameters_tensor.append([0.0, 0.0, 0.0])
            fit_tail_parameters_tensor.append([0.0, 0.0, 0.0])
            total_bsz.append(0)

        fit_head_parameters_tensor = torch.tensor(fit_head_parameters_tensor, dtype=torch.float32).cuda()
        fit_tail_parameters_tensor = torch.tensor(fit_tail_parameters_tensor, dtype=torch.float32).cuda()
        
        head_weights_tensor = []
        tail_weights_tensor = []
        total_weights = []
        
        for entity_id in range(len(entity2id)):
            head_weights_tensor.append([0.0, 0.0, 0.0])
            tail_weights_tensor.append([0.0, 0.0, 0.0])
            total_weights.append(0)
        
        head_weights_tensor = torch.tensor(head_weights_tensor, dtype=torch.float32).cuda()
        tail_weights_tensor = torch.tensor(tail_weights_tensor, dtype=torch.float32).cuda()
        
        
        
        
    

        # 初始化两个字典
        relation_head_entity_weight = {}
        relation_tail_entity_weight = {}

        

        
        
        #pdb.set_trace()
        # define the model
        kge_score = AdvMixRotatE(
            args=args,
            ent_tot=train_dataloader.get_ent_tot(),
            rel_tot=train_dataloader.get_rel_tot(),
            total_bsz = total_bsz,
            fit_head_parameters_tensor = fit_head_parameters_tensor,
            fit_tail_parameters_tensor = fit_tail_parameters_tensor,
            head_weights_tensor = head_weights_tensor,
            tail_weights_tensor = tail_weights_tensor,
            relation_head_entity_weight = relation_head_entity_weight,
            relation_tail_entity_weight = relation_tail_entity_weight,
            total_weights = total_weights,
            pt_exp_id = pt_exp_id,
            ft_exp_id = ft_exp_id,
            code_gen_file_path = code_gen_file_path,
            dim=args.dim,
            margin=args.margin,
            epsilon=2.0,
            img_emb=img_emb,
            text_emb=text_emb

        )
        #print(kge_score)
        # 将额外的张量添加到 state_dict 中
        # kge_score.state_dict()['fit_head_parameters_tensor'] = fit_head_parameters_tensor
        # kge_score.state_dict()['fit_tail_parameters_tensor'] = fit_tail_parameters_tensor

        
        
        logger.info(kge_score)
        # pdb.set_trace()
        # define the loss function
        model = NegativeSampling(
            model=kge_score,
            loss=SigmoidLoss(adv_temperature=args.adv_temp),
            batch_size=train_dataloader.get_batch_size(),
        )

        # train the model
        trainer = Trainer(
            args=args,
            logger=logger,
            model=model,
            data_loader=train_dataloader,
            train_times=args.epoch,
            alpha=args.learning_rate,
            use_gpu=True,
            opt_method='Adam',
            train_mode='normal',
            save_steps=100,
            checkpoint_dir=args.save
        )

        #pdb.set_trace()
        trainer.run()
        #pdb.set_trace()
        save_dir = f"{args.save}-MRR{trainer.Loss_log.get_acc()}"
        if not osp.exists(save_dir):
            # kge_score.save_checkpoint(save_dir)
            torch.save(trainer.best_model_wts, f"{args.save}-MRR{trainer.Loss_log.get_acc()}")
        
        
        
        #add, save chenckpoint
        kge_score.save_checkpoint(pt_checkpoint_dir)
        
        print("save checkpoint")

        # test the model
        #kge_score.load_checkpoint(save_dir)
        kge_score.load_checkpoint(pt_checkpoint_dir)
        #pdb.set_trace()
        tester = Tester(model=kge_score, data_loader=test_dataloader, use_gpu=True)
        mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain=False)
        logger.info(f"mrr:{mrr},\t mr:{mr},\t hit10:{hit10},\t hit3:{hit3},\t hit1:{hit1}")
        logger.info(f"{mrr}\t{mr}\t{hit10}\t{hit3}\t{hit1}")
        logger.info(" -------------------- finish! -------------------- ")
    else:
        # finetune 
        bsz = 128
        train_data = AnalogyFinetuneDataset(f'{file_path}/train2id_ft.txt', entity2id, relation2id)
        valid_data = AnalogyFinetuneDataset(f'{file_path}/valid2id_ft.txt', entity2id, relation2id)
        test_data = AnalogyFinetuneDataset(f'{file_path}/test2id_ft.txt', entity2id, relation2id) #_original

        #pdb.set_trace()

        train_dataloader = DataLoader(train_data, batch_size=bsz, shuffle=True, num_workers=4)
        valid_dataloader = DataLoader(valid_data, batch_size=bsz, shuffle=False, num_workers=4)
        test_dataloader = DataLoader(test_data, batch_size=bsz, shuffle=False, num_workers=4)


        #pdb.set_trace()


        #pdb.set_trace()

        total_bsz =[]

        # 为每个relation ID从1到192设置初始参数为空
        for relation_id in range(len(relation2id)):
            total_bsz.append(0)
            
            
        head_weights_tensor = []
        tail_weights_tensor = []
        total_weights = []
        
        for entity_id in range(len(entity2id)):
            head_weights_tensor.append([0.0, 0.0, 0.0])
            tail_weights_tensor.append([0.0, 0.0, 0.0])
            total_weights.append(0)
        
        head_weights_tensor = torch.tensor(head_weights_tensor, dtype=torch.float32).cuda()
        tail_weights_tensor = torch.tensor(tail_weights_tensor, dtype=torch.float32).cuda()
        
        # 初始化两个字典
        relation_head_entity_weight = {}
        relation_tail_entity_weight = {}
        
        
        #pdb.set_trace()
        
        # define the model
        model = AdvMixRotatE(
            args=args,
            ent_tot=len(entity2id),
            rel_tot=len(relation2id),
            total_bsz = total_bsz,
            fit_head_parameters_tensor = None,
            fit_tail_parameters_tensor = None,
            head_weights_tensor = head_weights_tensor,
            tail_weights_tensor = tail_weights_tensor,
            relation_head_entity_weight = relation_head_entity_weight,
            relation_tail_entity_weight = relation_tail_entity_weight,
            total_weights = total_weights,
            pt_exp_id = pt_exp_id,
            ft_exp_id = ft_exp_id,
            code_gen_file_path = code_gen_file_path,
            dim=args.dim,
            margin=args.margin,
            epsilon=2.0,
            img_emb=img_emb,
            text_emb=text_emb
            
        )

        
        exp_name = f"{args.dataset}"
        
        #pdb.set_trace()
        model.load_checkpoint(pt_checkpoint_dir)
        #pdb.set_trace()
        model.head_weights_tensor = torch.nn.Parameter(torch.zeros((len(entity2id), 3), dtype=torch.float32, device='cuda'), requires_grad=False)
        model.tail_weights_tensor = torch.nn.Parameter(torch.zeros((len(entity2id), 3), dtype=torch.float32, device='cuda'), requires_grad=False)
        
        
        checkpoint = torch.load(pt_checkpoint_dir)
        for key in checkpoint.keys():
            print(f"{key}: {type(checkpoint[key])}")

        #pdb.set_trace()

        # train the model
        trainer = Trainer(
            args=args,
            logger=logger,
            model=model,
            data_loader=train_dataloader,
            train_times=int(args.epoch/2),
            alpha=args.learning_rate,
            use_gpu=True,
            opt_method='Adagrad',
            train_mode='normal',
            save_steps=100,
            checkpoint_dir=args.save
            #finetune=True,
        )
        trainer.run()
        model.save_checkpoint(ft_checkpoint_dir)
        
        print("save checkpoint")
        # test the model
        model.load_checkpoint(ft_checkpoint_dir)
        
        
        
        # checkpoint = torch.load('ckpt/analogy/pt_Analogy_ANAG.ckpt')
        # for key in checkpoint.keys():
        #     print(f"{key}: {type(checkpoint[key])}")
        # pdb.set_trace()
        tester = Tester(model=model, data_loader=test_dataloader, use_gpu=True)
        mrr, mr, hit10, hti5, hti3, hit1 = tester.run_analogical_reasoning(type_constrain=False)
        print("mrr: ", mrr)
        print("mr: ", mr)
        print("hit10: ", hit10)
        print("hti5: ", hti5)
        print("hti3: ", hti3)
        print("hit1: ", hit1)
        
        
        
        
