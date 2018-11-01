import org.junit.jupiter.api.Test;

import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.database.Sequence;
import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.predictor.CPT.CPT.CPTPredictor;
import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.predictor.CPT.CPTPlus.CPTPlusPredictor;
import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.predictor.DG.DGPredictor;
import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.predictor.LZ78.LZ78Predictor;
import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.predictor.Markov.MarkovAllKPredictor;
import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.predictor.Markov.MarkovFirstOrderPredictor;
import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.predictor.TDAG.TDAGPredictor;

public class MixedPredictorTest extends PredictorTestUtils {

  // @formatter:off
	int[][] trainingSet = {
		{ 1, 1, 2, 1, 1, 1, 3 },
		{ 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 2, 1, 1, 1, 3 },
		{ 1, 1, 1, 1, 1, 1, 1 }
	};
  // @formatter:on
  int[] testSeq = { 2, 1, 1, 1 };

  @Test
  public void test_DGPredictor() {
    DGPredictor predictor = new DGPredictor("DG", "lookahead:4");
    predictor.Train(modelOf(trainingSet));
    Sequence resSeq = predictor.Predict(seqOf(0, testSeq));
    test(resSeq, 1);
  }

  @Test
  public void test_TDAGPredictor() {
    TDAGPredictor predictor = new TDAGPredictor();
    predictor.Train(modelOf(trainingSet));
    Sequence resSeq = predictor.Predict(seqOf(0, testSeq));
    test(resSeq, 3);
  }

  @Test
  public void test_CPTPlusPredictor() {
    CPTPlusPredictor predictor = new CPTPlusPredictor("CPT+",
        "CCF:true CBS:true CCFmin:1 CCFmax:6 CCFsup:2 splitMethod:0 splitLength:4 minPredictionRatio:1.0 noiseRatio:1.0");
    predictor.Train(modelOf(trainingSet));
    Sequence resSeq = predictor.Predict(seqOf(0, testSeq));
    test(resSeq, 1);
  }

  @Test
  public void test_CPTPredictor() {
    CPTPredictor predictor = new CPTPredictor();
    predictor.Train(modelOf(trainingSet));
    Sequence resSeq = predictor.Predict(seqOf(0, testSeq));
    test(resSeq, 1);
  }

  @Test
  public void test_MarkovFirstOrderPredictor() {
    MarkovFirstOrderPredictor predictor = new MarkovFirstOrderPredictor();
    predictor.Train(modelOf(trainingSet));
    Sequence resSeq = predictor.Predict(seqOf(0, testSeq));
    test(resSeq, 1);
  }

  @Test
  public void test_MarkovAllKPredictor() {
    MarkovAllKPredictor predictor = new MarkovAllKPredictor();
    predictor.Train(modelOf(trainingSet));
    Sequence resSeq = predictor.Predict(seqOf(0, testSeq));
    test(resSeq, 1);
  }

  @Test
  public void test_LZ78Predictor() {
    LZ78Predictor predictor = new LZ78Predictor();
    predictor.Train(modelOf(trainingSet));
    Sequence resSeq = predictor.Predict(seqOf(0, testSeq));
    test(resSeq, 1);
  }
}