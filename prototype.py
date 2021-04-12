from data import *
from lib import  *
import time

def proto_update(model):
    start_time = time.time()
    available_cls = []
    h_dict = {}
    feat_dict = {}
    missing_cls = []
    after_softmax_numpy_for_emergency = []
    feature_numpy_for_emergency = []
    max_prototype_bound = 100    
    model.eval()
    feature_extractor = model.roberta
    classifier_t = model.classifier

    for cls in range(len(source_classes)):
        h_dict[cls] = []
        feat_dict[cls] = []

    for batch in target_train_dl:
        input_ids = batch[0]
        attention_mask = batch[1]
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        fc1_lbcorr = feature_extractor.forward(input_ids, attention_mask)[0]
        
        #dropout 일단 생략  
        logits = classifier_t.forward(fc1_lbcorr)
        logits = logits.view(-1,65)
        m = nn.Softmax(dim=1)
        after_softmax = m(logits)
        fc1_lbcorr = fc1_lbcorr.view(-1, 768)
        
        pseudo_label = torch.argmax(after_softmax, dim=1)
        pseudo_label = pseudo_label.cpu()
        
        entropy = torch.sum(-after_softmax*torch.log(after_softmax+1e-10), dim=1, keepdim=True)
        entropy_norm = entropy / np.log(after_softmax.size(1))
        entropy_norm = entropy_norm.squeeze(1)
        entropy_norm = entropy_norm.cpu()

        for cls in range(len(source_classes)):
            # stack H for each class
            cls_filter = (pseudo_label == cls)
            list_loc = (torch.where(cls_filter == 1))[0]
            num_element = list(list_loc.data.numpy())
            if len(list_loc) == 0:
                missing_cls.append(cls)
                continue
            available_cls.append(cls)
                    
            filtered_ent = torch.gather(entropy_norm, dim=0, index=list_loc)            
            filtered_feat = torch.gather(fc1_lbcorr.cpu(), dim=0, index=list_loc.unsqueeze(1).repeat(1, 768))

            h_dict[cls].append(filtered_ent.cpu().data.numpy())
            feat_dict[cls].append(filtered_feat.cpu().data.numpy())

    available_cls = np.unique(available_cls)

    prototype_memory = []
    prototype_memory_dict = {}
    
    max_top1_ent = 0
    for cls in available_cls:
        ents_np = np.concatenate(h_dict[cls], axis=0)
        ent_idxs = np.argsort(ents_np)
        top1_ent = ents_np[ent_idxs[0]]
        if max_top1_ent < top1_ent:
            max_top1_ent = top1_ent
            max_top1_class = cls

    class_protypeNum_dict = {}
    max_prototype = 0

    for cls in available_cls:
        ents_np = np.concatenate(h_dict[cls], axis=0)
        ents_np_filtered = (ents_np <= 0.1)
        if ents_np_filtered.sum() == 0:
            class_protypeNum_dict[cls] = 1
        else:
            class_protypeNum_dict[cls] = ents_np_filtered.sum()

        if max_prototype < ents_np_filtered.sum():
            max_prototype = ents_np_filtered.sum()

        if max_prototype > 100:
            max_prototype = max_prototype_bound

    for cls in range(len(source_classes)):

        if cls in available_cls:
            ents_np = np.concatenate(h_dict[cls], axis=0)
            feats_np = np.concatenate(feat_dict[cls], axis=0)
            ent_idxs = np.argsort(ents_np)
            
            truncated_feat = feats_np[ent_idxs[:class_protypeNum_dict[cls]]]

            fit_to_max_prototype = np.concatenate([truncated_feat] * (int(max_prototype / truncated_feat.shape[0]) + 1),  axis=0)
            fit_to_max_prototype = fit_to_max_prototype[:max_prototype, :]
            
            prototype_memory.append(fit_to_max_prototype)
            prototype_memory_dict[cls] = fit_to_max_prototype
            
            
            
    print("** APM update... time:", time.time() - start_time)
    prototype_memory = np.concatenate(prototype_memory, axis=0)
    num_prototype_ = int(max_prototype)
    
    return prototype_memory, num_prototype_, prototype_memory_dict, available_cls 
