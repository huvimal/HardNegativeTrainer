import numpy as np
from common import *

def run():
    import torch
    torch.manual_seed(72);torch.set_num_threads(2)
    docs,queries=retrieval_fixture();baseline=encode(docs+queries)
    d=torch.tensor(baseline[:12],dtype=torch.float32);q=torch.tensor(baseline[12:],dtype=torch.float32)
    head=torch.nn.Linear(d.shape[1],32,bias=False);opt=torch.optim.Adam(head.parameters(),lr=.01)
    train_ids=list(range(8));test_ids=list(range(8,12));negative_log=[]
    for i in train_ids:
        order=np.argsort(-(baseline[12+i]@baseline[:12].T));negative_log.append({'query':i,'hard_negative':int(next(j for j in order if j!=i and j<8))})
    for step in range(80):
        loss=0
        for pair in negative_log:
            i=pair['query'];j=pair['hard_negative'];qe=torch.nn.functional.normalize(head(q[i:i+1]),dim=-1)
            de=torch.nn.functional.normalize(head(d[[i,j]]),dim=-1);logits=qe@de.T/.1
            loss+=torch.nn.functional.cross_entropy(logits,torch.tensor([0]))
        opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():learned=torch.nn.functional.normalize(head(q),dim=-1)@torch.nn.functional.normalize(head(d),dim=-1).T
    before=baseline[12:]@baseline[:12].T
    def hit(scores,ids):return float(np.mean(np.argmax(scores,axis=1)[ids]==np.array(ids)))
    return {'pretrained_test_hit1':hit(before,test_ids),'projection_test_hit1':hit(learned.numpy(),test_ids),
            'train_hit1':hit(learned.numpy(),train_ids),'hard_negative_pairs':negative_log,
            'checks':[check(not set(train_ids)&set(test_ids),'Training query IDs do not include evaluation IDs'),check(float(loss)>0,'Actual contrastive objective optimized')],
            'scope':'Train shared 32-dimensional bi-encoder projection heads on frozen pretrained MiniLM features. Transformer backbone is not fine-tuned. Tiny authored fixture, four evaluation topics.'}
