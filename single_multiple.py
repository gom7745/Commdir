#!/usr/bin/env python2
from os import system,chdir
dname = {'a9a':'a9a','covtype.libsvm.binary':'cov','epsilon_normalized':'eps','kddb':'kddb','news20.binary':'news','rcv1_test.binary':'rcvt','url_combined':'urlcombined','webspam_wc_normalized_trigram.svm':'webspam'}
f1 = open('figures/single_multiple_time.tex','w')
f2 = open('figures/single_multiple_pass.tex','w')

#remove those data/loss parameters you are not interested in their results
data = ['a9a','covtype.libsvm.binary','epsilon_normalized','kddb','news20.binary','rcv1_test.binary','url_combined','webspam_wc_normalized_trigram.svm']
loss_param = [1e-3, 1, 1e+3]

cmd = "make -C other_solvers >/dev/null 2>/dev/null; make -C commdir>/dev/null 2>/dev/null; mkdir ./log ./model ./table 2>/dev/null; mkdir figures/tikz 2> /dev/null;"
system(cmd)

methods = [1,2]
method_name =['single_commdir','commdir']


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
		for method in methods:
			dp = 'log/%s.LRC%s.%s'%(d,c,method_name[method-1])
			try:
				tmp_data = open(dp,'r').readlines()
			except:
				traindata = 'data/'+ d
				model = 'model/%s.C%s.%s.model'%(d,c,method_name[method-1])
				cmd = ''
				if (method == 1):
					cmd = 'commdir/ntrain -e 1e-6 -l 1 -i 1 -c %s %s %s >> %s 2> /dev/null'%(c, traindata, model, dp)
				else:
					cmd = 'commdir/ntrain -e 1e-6 -l 1 -c %s %s %s >> %s 2> /dev/null'%(c, traindata, model, dp)
				system('echo \'%s\' > %s'%(cmd, dp))
				system(cmd)

				tmp_data = open(dp,'r').readlines()
		print>>f1, '\\plotcommdirfunc{%s.LRC%s}{\\%s}'%(d,c,dname[d])
		print>>f2, '\\plotcommdirpass{%s.LRC%s}{\\%s}'%(d,c,dname[d])
	print >>f1, '\\caption{Comparison between single inner iteration and multiple inner iterations variants of the common-directions method. We present training time (in log scale) of logistic regression with $C=%s$.}'%c
	print >>f2, '\\caption{Comparison between single inner iteration and multiple inner iterations variants of the common-directions method. We present data passes (in log scale) of logistic regression with $C=%s$.}'%c
	print >> f1, '\\end{figure}'
	print >> f2, '\\end{figure}'
print >> f1, '\\end{document}'
print >> f2, '\\end{document}'
f1.close()
f2.close()

import gentable
chdir('figures/')
cmd = 'pdflatex single_multiple_time.tex'
system(cmd)
cmd = 'pdflatex single_multiple_pass.tex'
system(cmd)
cmd = 'make -f single_multiple_time.makefile'
system(cmd)
cmd = 'make -f single_multiple_pass.makefile'
system(cmd)
cmd = 'pdflatex single_multiple_time.tex'
system(cmd)
cmd = 'pdflatex single_multiple_pass.tex'
system(cmd)
cmd = 'rm *aux *log *figlist *makefile tikz/*'
system(cmd)
