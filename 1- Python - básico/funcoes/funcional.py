#armazenar uma função em uma variável 
def soma(a,b):
    return a + b
def sub(a,b):
    return a - b

somar = soma  
print(somar(2,3))


def operacao_aritmetica(fn,op1, op2):
    return fn(op1,op2)

resultado = operacao_aritmetica(soma,13,48)
print(resultado)

resultado = operacao_aritmetica(sub,13,48)
print(resultado)

#como diminuir o tempo de processamneto 
# escolha de estruturas eficientes , imapacto no tempo de processamento 
# evitar estruturas aninhadas 
#evitar cópias desnecessárias de dados 

def soma_parcial(a):
    # processamento pesao 1 , 2 ,3  - 60s
    def concluir_soma(b):
        return a + b 
    return concluir_soma

#mais linhas
fn = soma_parcial(10)
resultado_final=fn(12)
print(resultado_final)

# unica linha
resultado_final = soma_parcial(10)(12)