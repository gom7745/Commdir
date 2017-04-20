#!/usr/bin/env python2.7
import os
import hashlib
from opt import *

def findkey(line, key):
    t = line.partition(key)[2].split()[0]
    i = 0
    for c in reversed(t):
        if c.isalpha():
            i+=1
        else:
            break
    t = t[:len(t)-i]
    return float(t)

def parse_newton(newton_fname):
    f = open(newton_fname)
    l = []
    it = 0
    neg = False
    iters = 0
    for line in f:
        if line.startswith('running'):
            if neg:
                print 'func <= than opt'
            return l
        if not line.startswith('iter'):
            continue
        func = findkey(line, 'f ')
        g = findkey(line, '|g| ')
        if (iters != 0):
            l.append((func, g, t, it))
        t = findkey(line, 'time ')
        cg = findkey(line, 'CG ')
        it = it + 2 + cg
        iters = iters + 1
        if l:
            t = t + l[-1][2]
    return l

def parse_commdir(commdir_fname):
    f = open(commdir_fname)
    l = []
    it = 0
    neg = False
    iters = 0
    for line in f:
        if len(line.strip()) == 0:
            continue
        if line.startswith('running'):
            return l
        if 'iter=' not in line:
            continue

        func = None
        try:
            func = findkey(line, ' f=')
        except:
            func = findkey(line, 'func=')
        g = findkey(line, '|g|=')
        if (iters != 0):
            l.append((func, g, t, it))
        iters = iters + 1
        t = findkey(line, 'time=')
        it = it + 2
        if l:
            t = t + l[-1][2]

    if neg:
        print 'func <= opt'

    return l

def parse_ag(ag_fname):
    f = open(ag_fname)
    l = []
    it = 0
    neg = False
    iters = 0
    for line in f:
        if len(line.strip()) == 0:
            continue
        if line.startswith('running'):
            return l
        if not line.startswith('iter='):
            continue

        func = None
        try:
            func = findkey(line, ' f=')
        except:
            func = findkey(line, 'func=')
        g = findkey(line, '|g|=')
        if (iters != 0):
            l.append((func, g, t, it))
        iters = iters + 1
        t = findkey(line, 'time=')
        linesearch = findkey(line, 'passes=')
        it = it + linesearch*2
        if l:
            t = t + l[-1][2]
        l.append((func, g, t, it))

    if neg:
        print 'func <= opt'

    return l

def parse_lbfgs(lbfgs_fname):
    f = open(lbfgs_fname)
    l = []
    it = 0
    neg = False
    for line in f:
        if len(line.strip()) == 0:
            continue
        if line.startswith('running'):
            return l
        if not line.startswith('iter='):
            continue

        func = None
        try:
            func = findkey(line, ' f=')
        except:
            func = findkey(line, 'func=')
        g = findkey(line, '|g|=')
        t = findkey(line, 'time=')
        it = it + 2
        l.append((func, g, t, it))

    if neg:
        print 'func <= opt'

    return l

def to_csv(l):
    csv = []
    csv.append('func, g, t, pass\n')
    for i in l:
        csv.append('{:.16g}, {:.16g}, {:.16g}, {:.16g}\n'.format(i[0], i[1], i[2], i[3]))
    return ''.join(csv)

def valid_file(x):
    solver_name = x.split('.')[-1]
    valid_start = ['commdir', 'lbfgs','bfgs','newton','ag','single_commdir']
    for s in valid_start:
        if solver_name.startswith(s):
            return True
    return False

logfiles = os.listdir('log')
logs = filter(valid_file, logfiles)

def relative_func_transform(parsed_log, optval):
    l = []
    lineno = 1
    for (func, g, t, it) in parsed_log:
        rel = (func-optval) / optval
        if rel < 0:
            print "Err: Rel < 0"
            print "\titer=%d"%lineno, "rel=%.14e"%rel, "f=%.14e"%func, "optval=%.14e"%optval

        l.append((rel, g, t, it))
        lineno = lineno + 1
    return l

answers = filter(lambda x: x.endswith('.ans'), logfiles)

for logname in logs:
    try:
        t = logname.split('.')
        data_name = '.'.join(t[:-1])
        solver_name = t[-1]
    except Exception:
        print 'Err file name', logname
        continue

    optval = None
    try:
        optval = opt[data_name]
    except KeyError:
        print 'Err: Could not find opt val for', data_name, logname
        print
        continue

    #  print data_name, solver_name, optval

    parsed_log = None
    if solver_name.startswith('newton'):
        parsed_log = parse_newton('log/'+logname)
    elif solver_name.startswith('commdir') or solver_name.startswith('single_commdir'):
        parsed_log = parse_commdir('log/'+logname)
    elif solver_name.startswith('ag'):
        parsed_log = parse_ag('log/'+logname)
    elif solver_name.startswith('dcd'):
        parsed_log = parse_dcd('log/'+logname)
    elif solver_name.startswith('lbfgs') or solver_name.startswith('bfgs'):
        parsed_log = parse_lbfgs('log/'+logname)
    else:
        continue
        #raise Exception

    if not parsed_log:
        print 'log err with', solver_name, logname
        continue

    parsed_log = relative_func_transform(parsed_log, optval)

    csv_name = 'table/'+logname
    new_csv_log = to_csv(parsed_log)
    try:
        old_csv_log = open(csv_name).read()
    except IOError:
        old_csv_log = ''

    def digest(s):
        m = hashlib.md5()
        m.update(s)
        return m.digest()

    new_md5 = digest(new_csv_log)
    old_md5 = digest(old_csv_log)
    if new_md5 != old_md5:
        f = open(csv_name, 'w')
        f.write(new_csv_log)
        print 'update', csv_name
