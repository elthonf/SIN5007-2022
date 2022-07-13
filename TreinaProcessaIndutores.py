import unittest


class MyTestAvulso(unittest.TestCase):
    def Atest_processaAvulso(self):
        from processaBase import ProcessContent
        from processaBase import ProcessaSVM

        #processa = ProcessaSVM(mytype=ProcessContent.ALL)
        #processa.processa_plano_id(0)

        for a in [ProcessContent.ALL, ProcessContent.PCA, ProcessContent.SELECT01, ProcessContent.SELECT02]:
            processa = ProcessaSVM(mytype=a)
            planos = processa.generate_plano()
            for i, p in enumerate(planos): #for i in [32, 31, 30, 29, 28]: #for i, p in enumerate(planos):
                processa = ProcessaSVM(mytype=a)
                processa.processa_plano_id(plano_id=i)

        pass

class MyRunSVM(unittest.TestCase):
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

        #processa = ProcessaSVM(mytype=ProcessContent.ALL)
        #processa.processa_plano_id(0)

        for a in [ProcessContent.ALL]:
            processa = ProcessaSVM(mytype=a)
            planos = processa.generate_plano()
            for i, p in enumerate(planos):
            #for i in list(range(11, 9, -1)):
                processa = ProcessaSVM(mytype=a)
                processa.processa_plano_id(plano_id=i)

        pass

class MyRunDeepLearning(unittest.TestCase):
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

class MyRunRandomForest(unittest.TestCase):
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
