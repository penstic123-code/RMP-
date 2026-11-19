import numpy as np
D,Y,T,L,Q=96,8,1024,16,4
F=np.float32
def run(seed, rule, search,Q=4):
    rg=np.random.default_rng(seed)
    pools=np.stack([rg.choice(D,3,False) for _ in range(Y)])
    dense=search=='dense'
    idx=pools.astype('i') if search=='known' else np.zeros((Y,3),'i')
    w=np.zeros((Y,3),F)
    used=np.full(Y,3 if search=='known' else 0,'i')
    st=[w,idx,used]
    if dense:
        w=np.zeros((Y,D),F);st=[w];del idx,used
    if rule=='rls':
        cov=np.tile(4*np.eye(3,dtype=np.float64),(Y,1,1));st.append(cov)
    rng=np.random.default_rng(seed+1337)
    shared=search=='shared'
    if search=='window':
        hist=np.zeros((L,D),F); residual=np.zeros((L,Y),F);st.extend([hist,residual])
    elif search not in ('known','dense'):
        cand=np.zeros((1 if shared else Y,Q),'i')
        xx=np.zeros_like(cand,dtype=F)
        xy=np.zeros((Y,Q),F)
        ee=np.zeros(Y,F)
        st.extend([cand,xx,xy,ee])
    byphase=[]
    for phase in range(3):
        losses=[]
        for t in range(T):
            x=rg.standard_normal(D).astype(F)
            y=.8*x[pools[:,phase]]+rg.normal(0,.35,Y)
            v=x[None,:] if dense else x[idx]*(np.arange(3)<used[:,None])
            e=(y-(w*v).sum(1)).astype(F)
            losses.append(np.mean(e*e))
            if rule=='lms': w+=.04*e[:,None]*v
            if rule=='nlms': w+=.15*e[:,None]*v/(1+(v*v).sum(1,keepdims=True))
            if rule=='rls':
                pv=(cov@v[:,:,None])[:,:,0]
                gain=pv/(.98+(v*pv).sum(1,keepdims=True))
                w+=gain*e[:,None]
                cov[:]=(cov-gain[:,:,None]*pv[:,None,:])/.98
                cov[:]=(cov+cov.transpose(0,2,1))*.5
            if search in ('known','dense'): continue
            if search=='window':
                hist[t%L]=x;residual[t%L]=e
                if (t+1)%L: continue
                cand=np.arange(D)[None,:]
                score=np.abs(residual.T@hist)/np.sqrt(.001+(residual**2).sum(0)[:,None]*(hist**2).sum(0))
            else:
                if t%L==0:
                    for row in cand: row[:]=rng.choice(D,Q,False)
                    xx.fill(0);xy.fill(0);ee.fill(0)
                vals=x[cand];xx+=vals*vals;xy+=vals*e[:,None];ee+=e*e
                if (t+1)%L: continue
                score=np.abs(xy)/np.sqrt(.001+xx*ee[:,None])
            for k in range(Y):
                j=np.argmax(score[k]);src=cand[0 if shared or search=="window" else k,j]
                if score[k,j]<.78 or src in idx[k,:used[k]]: continue
                slot=used[k] if used[k]<3 else np.argmin(abs(w[k]))
                idx[k,slot]=src;w[k,slot]=0;used[k]=min(3,used[k]+1)
        byphase.append([np.mean(losses[:256]),np.mean(losses[-256:])])
    return np.mean(byphase[1:],axis=0),sum(a.nbytes for a in st)
if __name__=='__main__':
    for split,seeds in [('dev',range(3)),('explore',range(30,40)),('confirm',range(100,110))]:
        for rule,search,q in [('lms','known',4),('nlms','known',4),('rls','known',4),('nlms','dense',4),('nlms','window',4),('nlms','independent',4),('nlms','shared',4),('nlms','shared',16),('nlms','shared',32)]:
            vals=[run(s,rule,search,q) for s in seeds]
            print(split,rule,search,q,np.round(np.mean([v[0] for v in vals],axis=0),4),vals[0][1],flush=True)
