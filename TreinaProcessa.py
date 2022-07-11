import unittest


class MyTestCase(unittest.TestCase):
    def Atest_processa01(self):
        from main import ProcessContent
        from main import ProcessaSVM

        processa = ProcessaSVM(mytype=ProcessContent.SELECT01)
        processa.processa_plano_id(60)
        pass

        for a in range(32):
            print( f"python main.py 1 {a} & ")
            if (a+1) % 6 == 0:
                print(f"wait\n")
    def test_processaRandomForest(self):
        from main import ProcessContent
        from main import ProcessaRandomForest

        #processa = ProcessaRandomForest(mytype=ProcessContent.ALL)
        #processa.processa_plano_id(0)

        for a in [ProcessContent.ALL, ProcessContent.PCA, ProcessContent.SELECT01, ProcessContent.SELECT02]:
            processa = ProcessaRandomForest(mytype=a)
            planos = processa.generate_plano()
            for i, p in enumerate(planos):
                processa = ProcessaRandomForest(mytype=a)
                processa.processa_plano_id(plano_id=i)

        pass

if __name__ == '__main__':
    unittest.main()
