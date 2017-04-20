#!/usr/bin/env python2
from os import system,chdir
dname = {'a9a':'a9a','covtype.libsvm.binary':'cov','epsilon_normalized':'eps','kddb':'kddb','news20.binary':'news','rcv1_test.binary':'rcvt','url_combined':'urlcombined','webspam_wc_normalized_trigram.svm':'webspam'}

f1 = open('figures/methods_bias_time.tex','w')
f2 = open('figures/methods_bias_pass.tex','w')
#remove those method/data/loss parameters you are not interested in its result
methodlist = ['lbfgs30','bfgs','newton','ag','commdir']
data = ['a9a','covtype.libsvm.binary','epsilon_normalized','kddb','news20.binary','rcv1_test.binary','url_combined','webspam_wc_normalized_trigram.svm']
loss_param = [1e-3, 1, 1e+3]
solver = {'lbfgs30':0,'bfgs':0,'newton':1,'ag':2}

cmd = "make -C other_solvers>/dev/null 2>/dev/null; make -C commdir>/dev/null 2>/dev/null; mkdir ./log ./model ./table 2>/dev/null; mkdir figures/tikz 2> /dev/null;"
system(cmd)

print >> f1, '\\documentclass[twoside,11pt]{article}'
print >> f2, '\\documentclass[twoside,11pt]{article}'
print >> f1, '\\input{header.tex}'
print >> f2, '\\input{header.tex}'
print >> f1, '\\begin{document}'
print >> f2, '\\begin{document}'

for c in loss_param:
	print >> f1, '\\begin{figure}[ht]'
	print >> f2, '\\begin{figure}[ht]'

	for d in data:
		if c == 1e+3 and d == 'kddb':
			continue
		for method in methodlist:
			dp = 'log/%s.bias.LRC%s.%s'%(d,c, method)
			try:
				tmp_data = open(dp,'r').readlines()
			except:
				traindata = 'data/'+ d
				model = 'model/%s.bias.C%s.%s.model'%(d,c,method)
				cmd = ''
				m = 1000
				if (method == 'lbfgs30'):
					m = 30
				if (method != 'commdir'):
					cmd = 'other_solvers/train -B 1 -e 1e-6 -s %s -c %s -m %s %s %s > %s'%(solver[method], c, m, traindata, model, dp)

				else:
					cmd = 'commdir/ntrain -B 1 -e 1e-6 -l 1 -i 1 -c %s %s %s > %s 2>/dev/null'%(c, traindata, model, dp)
				system('echo \'%s\' >> %s'%(cmd, dp))
				system(cmd)

		print>>f1, '\\plotfunc{%s.bias.LRC%s}{\\%s}'%(d,c,dname[d])
		print>>f2, '\\plotpass{%s.bias.LRC%s}{\\%s}'%(d,c,dname[d])
	print >>f1, '\\caption{Training time of logistic regression with a bias term and $C=%s$.}'%c
	print >>f2, '\\caption{Number of data passes of logistic regression with a bias term and $C=%s$.}'%c
	print >> f1, '\\end{figure}'
	print >> f2, '\\end{figure}'
print >> f1, '\\end{document}'
print >> f2, '\\end{document}'
f1.close()
f2.close()

import gentable
chdir('figures/')
cmd = 'pdflatex methods_bias_time.tex'
system(cmd)
cmd = 'pdflatex methods_bias_pass.tex'
system(cmd)
cmd = 'make -f methods_bias_time.makefile'
system(cmd)
cmd = 'make -f methods_bias_pass.makefile'
system(cmd)
cmd = 'pdflatex methods_bias_time.tex'
system(cmd)
cmd = 'pdflatex methods_bias_pass.tex'
system(cmd)
cmd = 'rm *aux *log *figlist *makefile tikz/*'
system(cmd)
