#!/usr/bin/python3


from typing import List, Any

def intersperse(wedge: Any, ls: List[Any]) -> List[Any]:
    if not ls: return []
    return ls[:1] + prependAll(wedge, ls[1:])


def prependAll(wedge: Any, ls: List[Any]) -> List[Any]:
    if not ls: return []
    out = []
    for x in ls:
        out.append(wedge)
        out.append(x)
    return out



def main():
    f = open("com-lj.all.cmty.txt", 'r')
    outfile = open("out.dot", 'w')
    
    print("digraph graphname {\n", file=outfile)
    
    for line in f.readlines():
        nodes = line.split()
        print(' '.join(intersperse('->', nodes)), ';', file=outfile)
        
    print("\n}\n", file=outfile)
    print('done')
    



if __name__ == '__main__':
    main()
