'''
Created on 2015.2.15

@author: Yikang

Merge all training texts into a single text file. 
One sentence per line.
'''

import fileinput
import re
from glob import glob

fpath = 'data/aclImdb/train/all/'

if __name__ == '__main__':
    unaccepted_chars = re.compile(ur"[^\w,.?!()'\-\ ]+")
    final_chars = re.compile(ur"[.?!]+$")
    
    output = open(fpath.replace('/', '_')+'training_IMDB.txt','w')
    ncommand = 0
    for line in fileinput.input(glob(fpath+'*.txt')):
        line = line.strip()
        lines = line.split('<br /><br />')
        nsentence = 0
        for l in lines:
            l = l.lower()
            l = re.sub(unaccepted_chars, ' ', l)
            l = l.replace('-', ' ')
            l = l.replace('. ', '.\n')
            #l = l.replace(',', ' ,')
            l = l.replace('? ', '?\n')
            l = l.replace('! ', '!\n')
            ls = l.split('\n')
            s = ''
            for ll in ls:
                if ll.strip() != '':
                    #ll = ll.replace(',',' ,')
                    ll = re.sub(r'([,.?!()]+)', r' \1 ', ll)
                    while ll.find('  ') > -1:
                        ll = ll.replace('  ', ' ')
                    #output.write(fpath.replace('/', '_')+str(ncommand)+'_'+str(nsentence)+' ')
                    s += ll.strip() + ' '
                    nsentence += 1
            s = s.replace(' . ', ' .\n')
            s = s.replace(' ? ', ' ?\n')
            s = s.replace(' ! ', ' !\n')
            output.write(s.strip() + '\n')
        #output.write('\n')
        ncommand += 1
    fileinput.close()
    output.close()