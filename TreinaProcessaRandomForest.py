import unittest


class MyTestCase(unittest.TestCase):
    def Atest_processa01(self):
        from processaBase import ProcessContent
        from processaBase import ProcessaSVM

        processa = ProcessaSVM(mytype=ProcessContent.SELECT01)
        processa.processa_plano_id(60)
        pass

        for a in range(32):
            print( f"python main.py 1 {a} & ")
            if (a+1) % 6 == 0:
                print(f"wait\n")
    def test_processaRandomForest(self):
        from processaBase import ProcessContent
        from processaBase import ProcessaRandomForest

        #processa = ProcessaRandomForest(mytype=ProcessContent.ALL)
        #processa.processa_plano_id(0)

        for a in [ProcessContent.ALL, ProcessContent.PCA, ProcessContent.SELECT01, ProcessContent.SELECT02]:
            processa = ProcessaRandomForest(mytype=a)
            planos = processa.generate_plano()
            for i, p in enumerate(planos):
                processa = ProcessaRandomForest(mytype=a)
                processa.processa_plano_id(plano_id=i)

        pass

    def test_processaRandomForest3(self):
        from processaBase import ProcessContent
        from processaBase import ProcessaRandomForest

        #processa = ProcessaRandomForest(mytype=ProcessContent.ALL)
        #processa.processa_plano_id(0)

        for a in [ProcessContent.SELECT01]:
            processa = ProcessaRandomForest(mytype=a)
            planos = processa.generate_plano()
            for i, p in enumerate(planos):
                processa = ProcessaRandomForest(mytype=a)
                processa.processa_plano_id(plano_id=i)

        pass
    def test_processaRandomForest4(self):
        from processaBase import ProcessContent
        from processaBase import ProcessaRandomForest

        #processa = ProcessaRandomForest(mytype=ProcessContent.ALL)
        #processa.processa_plano_id(0)

        for a in [ProcessContent.SELECT02]:
            processa = ProcessaRandomForest(mytype=a)
            planos = processa.generate_plano()
            for i, p in enumerate(planos):
                processa = ProcessaRandomForest(mytype=a)
                processa.processa_plano_id(plano_id=i)

        pass

    def test_processaRandomForest2(self):
        from processaBase import ProcessContent
        from processaBase import ProcessaRandomForest

        #processa = ProcessaRandomForest(mytype=ProcessContent.ALL)
        #processa.processa_plano_id(0)

        for a in [ProcessContent.PCA]:
            processa = ProcessaRandomForest(mytype=a)
            planos = processa.generate_plano()
            for i, p in enumerate(planos):
                processa = ProcessaRandomForest(mytype=a)
                processa.processa_plano_id(plano_id=i)

        pass

    def test_processaRandomForest1(self):
        from processaBase import ProcessContent
        from processaBase import ProcessaRandomForest

        #processa = ProcessaRandomForest(mytype=ProcessContent.ALL)
        #processa.processa_plano_id(0)

        for a in [ProcessContent.ALL]:
            processa = ProcessaRandomForest(mytype=a)
            planos = processa.generate_plano()
            for i, p in enumerate(planos):
            #for i in list(range(11, 9, -1)):
                processa = ProcessaRandomForest(mytype=a)
                processa.processa_plano_id(plano_id=i)

        pass

if __name__ == '__main__':
    unittest.main()