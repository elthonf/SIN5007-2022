import unittest


class MyTestCase(unittest.TestCase):
    def Atest_processaSVM(self):
        from processaBase import ProcessContent
        from processaBase import ProcessaSVM

        #processa = ProcessaSVM(mytype=ProcessContent.ALL)
        #processa.processa_plano_id(0)

        for a in [ProcessContent.ALL, ProcessContent.PCA, ProcessContent.SELECT01, ProcessContent.SELECT02]:
            processa = ProcessaSVM(mytype=a)
            planos = processa.generate_plano()
            for i, p in enumerate(planos):
                processa = ProcessaSVM(mytype=a)
                processa.processa_plano_id(plano_id=i)

        pass

    def test_processaSVM3(self):
        from processaBase import ProcessContent
        from processaBase import ProcessaSVM

        #processa = ProcessaRandomForest(mytype=ProcessContent.ALL)
        #processa.processa_plano_id(0)

        for a in [ProcessContent.SELECT01]:
            processa = ProcessaSVM(mytype=a)
            planos = processa.generate_plano()
            for i, p in enumerate(planos):
                processa = ProcessaSVM(mytype=a)
                processa.processa_plano_id(plano_id=i)

        pass
    def test_processaSVM4(self):
        from processaBase import ProcessContent
        from processaBase import ProcessaSVM

        #processa = ProcessaRandomForest(mytype=ProcessContent.ALL)
        #processa.processa_plano_id(0)

        for a in [ProcessContent.SELECT02]:
            processa = ProcessaSVM(mytype=a)
            planos = processa.generate_plano()
            for i, p in enumerate(planos):
                processa = ProcessaSVM(mytype=a)
                processa.processa_plano_id(plano_id=i)

        pass

    def test_processaSVM2(self):
        from processaBase import ProcessContent
        from processaBase import ProcessaSVM

        #processa = ProcessaRandomForest(mytype=ProcessContent.ALL)
        #processa.processa_plano_id(0)

        for a in [ProcessContent.PCA]:
            processa = ProcessaSVM(mytype=a)
            planos = processa.generate_plano()
            for i, p in enumerate(planos):
                processa = ProcessaSVM(mytype=a)
                processa.processa_plano_id(plano_id=i)

        pass

    def test_processaSVM1(self):
        from processaBase import ProcessContent
        from processaBase import ProcessaSVM

        #processa = ProcessaRandomForest(mytype=ProcessContent.ALL)
        #processa.processa_plano_id(0)

        for a in [ProcessContent.ALL]:
            processa = ProcessaSVM(mytype=a)
            planos = processa.generate_plano()
            for i, p in enumerate(planos):
            #for i in list(range(11, 9, -1)):
                processa = ProcessaSVM(mytype=a)
                processa.processa_plano_id(plano_id=i)

        pass

if __name__ == '__main__':
    unittest.main()
