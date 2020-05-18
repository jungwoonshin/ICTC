ICTC is a bipartite link prediction method.\

To run the model, please first select one dataset from args.py while commenting out the ones that you are not using.\

You have three model choices. \

To run ICTC, python train.py. This will run GAE,LGAE,ICTC.\
To run BiSPM, python bispm.py \
To run SRNMF, choose one similarity measure which is one of [CN, JC,CPA] by selecting one from srnmf.py. (i.e. args.similarity = 'srnmf_cn') then python srnmf.py


