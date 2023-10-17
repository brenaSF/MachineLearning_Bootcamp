#função - recebe parâmetros(entrada) - processamento - saída
#sobrescrita
def saudacao(nome ='Pessoa'):
    print(f'Bom dia {nome}!')


def saudacao(nome = 'Pessoa', idade=20):
    print(f'Bom dia  {nome} \n Vc nem parece ter {idade} anos')

if __name__ == '__main__':
    saudacao(nome = 'Pessoa', idade = 20 )
print(__name__)

# quando é chamando o método pelo próprio programa, recebe o nome main (principal)
# caso seja chamando por outro programa , recebe o nome do pacote e do programa que está 
# escopo - variável definido apenas ao bloco associado a função

def some_e_multi(a,b,x):
    return a+b*x