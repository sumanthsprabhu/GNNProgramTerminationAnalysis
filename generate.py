import numpy as np 
import threading
import time 

#for i in {1..30}; python3.8 generate.py; done

def recInit(n, lines) :
    x = np.random.randint(0,20)
    y = np.random.randint(0,20)

    lines.append(f'{" "*n}import time')
    lines.append(f'{" "*n}')
    lines.append(f'{" "*n}x = {x}')
    lines.append(f'{" "*n}y = {y}')

    return 0

def recNewAssign(n, lines) :

    x = np.random.randint(0,20)
    y = np.random.randint(0,20)

    if np.random.rand() > 0.5 : 
        lines.append(f'{" "*n}x = {x}')
    else : 
        lines.append(f'{" "*n}y = {y}')

    return 0

def recInc(n, lines) : 
    if np.random.rand() > 0.5 : 
        lines.append(f'{" "*n}x +=1')
    else : 
        lines.append(f'{" "*n}x -=1')

    return 0
    
def recWhile(n, lines) : 

    if np.random.rand() > 0.5 : 
        lines.append(f'{" "*n}while x > y : ')
    else : 
        lines.append(f'{" "*n}while y > x : ')

    return 1

def recSleep(n, lines) : 
    lines.append(f'{" "*n}time.sleep(2)')

    return 0

def recFor(n, lines) : 
    x = np.random.randint(0,20)
    lines.append(f'{" "*n}for i in range({x}) : ')

    return 1

def recPrint(n, lines) :     
    lines.append(f'{" "*n}#print("test:", x)')

    return 0

def recComment(n, lines) :     
    lines.append(f'{" "*n}#debug test')

    return 0

def recIf(n, lines) : 
    x = np.random.randint(0,20)
    
    if np.random.rand() > 0.5 :
        lines.append(f'{" "*n}if x < {x} : ')
    else : 
        lines.append(f'{" "*n}if y > {x} : ')

    return 1

def generateCode() : 
    
    my_list = [recSleep, recWhile, recFor, recInc, recNewAssign, recIf]

    lines = []
    
    recInit(0, lines)    
    n = 0
    for i in range(10) : 
        lb = np.random.choice(my_list)(n, lines)
        if lb : 
            n += 2
        else : 
            if np.random.rand() < 0.2 and n >1: 
                n-= 2
    recInc(n, lines)

    #for line in lines : 
    #    print(line)

    code = '\n'.join(lines)
    return code    

def testExec(code) : 

    try :         
        exec(code)
        #print('passed')        
        return 1
    except Exception as e :         
        #print('failed')
        print(e)
        return 0    
    

def generateItem() : 

    code = generateCode()
    start = time.time()    
    term = 0

    t1 = threading.Thread(target=testExec, args =(code,), daemon=True)
    t1.start()

    while time.time() - start < 3 :         
        time.sleep(0.1)        
        if not t1.is_alive() : 
            term = 1
            break

    return code, term

#code, term = generateItem()
#print('done')

data = []
labels = []
for i in range(10) : 
    code, term = generateItem()

    ran = np.random.randint(0,60000) 
    print(f'{ran} {term}')
    data.append(code)
    labels.append(term)
    
    if term :         
        with open(f'dataset/1/{ran}', 'w') as f: 
            f.write(code)
    else :         
        with open(f'dataset/0/{ran}', 'w') as f: 
            f.write(code)

