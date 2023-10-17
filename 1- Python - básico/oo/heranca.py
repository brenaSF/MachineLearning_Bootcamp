# contador da classe e o instanciado são diferentes 

class Contador : 
    contador = 10  #atributo da class

    def inc_maluco(self):
        self.contador +=1
        return self.contador
    
    @classmethod
    def inc(cls):
        cls.contador += 1
        return cls.contador
    
    @classmethod # método de classe - não precidade de instância
    def dec (cls):
        cls.contador -= 1 
        return cls.contador
    
    @staticmethod
    def mais_um(n):
        return n + 1 

c1 = Contador()
print(c1.inc_maluco())
print(c1.inc_maluco())
print(c1.inc_maluco())
print(c1.inc_maluco())


print(Contador.inc())
print(Contador.inc())
print(Contador.inc())

print(Contador.dec())
print(Contador.dec())
print(Contador.dec())


print(Contador.mais_um(99))

