import unittest


class MyTestCase(unittest.TestCase):
    def test_processaDeepLearning(self):
        from processaBase import ProcessContent
        from processaBase import ProcessaDeepLearning

        #processa = ProcessaRandomForest(mytype=ProcessContent.ALL)
        #processa.processa_plano_id(0)

        for a in [ProcessContent.ALL, ProcessContent.PCA, ProcessContent.SELECT01, ProcessContent.SELECT02]:
            processa = ProcessaDeepLearning(mytype=a)
            planos = processa.generate_plano()
            for i, p in enumerate(planos):
                processa = ProcessaDeepLearning(mytype=a)
                processa.processa_plano_id(plano_id=i)

        pass

    def test_processaDeepLearning3(self):
        from processaBase import ProcessContent
        from processaBase import ProcessaDeepLearning

        #processa = ProcessaRandomForest(mytype=ProcessContent.ALL)
        #processa.processa_plano_id(0)

        for a in [ProcessContent.SELECT01]:
            processa = ProcessaDeepLearning(mytype=a)
            planos = processa.generate_plano()
            for i, p in enumerate(planos):
                processa = ProcessaDeepLearning(mytype=a)
                processa.processa_plano_id(plano_id=i)

        pass
    def test_processaDeepLearning4(self):
        from processaBase import ProcessContent
        from processaBase import ProcessaDeepLearning

        #processa = ProcessaRandomForest(mytype=ProcessContent.ALL)
        #processa.processa_plano_id(0)

        for a in [ProcessContent.SELECT02]:
            processa = ProcessaDeepLearning(mytype=a)
            planos = processa.generate_plano()
            for i, p in enumerate(planos):
                processa = ProcessaDeepLearning(mytype=a)
                processa.processa_plano_id(plano_id=i)

        pass

    def test_processaDeepLearning2(self):
        from processaBase import ProcessContent
        from processaBase import ProcessaDeepLearning

        #processa = ProcessaRandomForest(mytype=ProcessContent.ALL)
        #processa.processa_plano_id(0)

        for a in [ProcessContent.PCA]:
            processa = ProcessaDeepLearning(mytype=a)
            planos = processa.generate_plano()
            for i, p in enumerate(planos):
                processa = ProcessaDeepLearning(mytype=a)
                processa.processa_plano_id(plano_id=i)

        pass

    def test_processaDeepLearning1(self):
        from processaBase import ProcessContent
        from processaBase import ProcessaDeepLearning

        #processa = ProcessaRandomForest(mytype=ProcessContent.ALL)
        #processa.processa_plano_id(0)

        for a in [ProcessContent.ALL]:
            processa = ProcessaDeepLearning(mytype=a)
            planos = processa.generate_plano()
            for i, p in enumerate(planos):
            #for i in list(range(11, 9, -1)):
                processa = ProcessaDeepLearning(mytype=a)
                processa.processa_plano_id(plano_id=i)

        pass

if __name__ == '__main__':
    unittest.main()
