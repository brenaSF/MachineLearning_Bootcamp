#decorator @property

#classes, objetos , init(inicializar objeto)
#indicar o parâmetro - padrão (caso não seja informado)


class Produto:
    def __init__(self,nome,preco=1.99,desc=0):
        self.nome = nome
        self.__preco = preco ##privado
        self.desc = desc

    @property
    def preco(self):
        return self.__preco
    
    @preco.setter
    def preco(self,novo_preco):
        if novo_preco > 0:
          self.__preco = novo_preco

    @property #variável e não método    
    def preco_final(self):
        return (1 - self.desc) * self.preco
    


p1 = Produto('Caneta',3.89,19.56)
p2 = Produto('Caderno',14,0.2) # Produto.__init__(p1,...)

p1.preco = 70.89
p1.preco = 45.89 #ler - property setter - alterar


#self.__nome
#p1.__Produto__nome - atributo privado 

print(p1.nome,p1.preco,p1.desc,p1.preco_final)
print(p2.nome,p2.preco,p2.desc,p2.preco_final)