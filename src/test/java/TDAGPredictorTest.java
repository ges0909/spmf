import org.junit.jupiter.api.Test;

import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.database.Sequence;
import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.predictor.TDAG.TDAGPredictor;

public class TDAGPredictorTest extends PredictorTestUtils {

  private Sequence predict(int[][] trainingSet, int[] testSeq) {
    TDAGPredictor predictor = new TDAGPredictor();
    predictor.Train(modelOf(trainingSet));
    return predictor.Predict(seqOf(0, testSeq));
  }

  @Test
  public void training_A() {
    // @formatter:off
		int[][] trainingSet = {
			{ 1, 1 },
			{ 1, 1 }
		};
		// @formatter:on
    test(predict(trainingSet, new int[] { 1 }), 1);
  }

  @Test
  public void training_B() {
    // @formatter:off
		int[][] trainingSet = {
			{ 1, 2 },
			{ 1, 2 }
		};
		// @formatter:on
    test(predict(trainingSet, new int[] { 1 }), 2);
  }

  @Test
  public void training_C() {
    // @formatter:off
		int[][] trainingSet = {
			{ 1, 1, 1 },
			{ 1, 1, 1 }
		};
		// @formatter:on
    test(predict(trainingSet, new int[] { 1 }), 1);
  }

  @Test
  public void training_D() {
    // @formatter:off
		int[][] trainingSet = {
			{ 1, 1, 1, 2 },
			{ 1, 1, 1, 2 }
		};
		// @formatter:on
    test(predict(trainingSet, new int[] { 1 }), 1);
  }

  @Test
  public void training_E() {
    // @formatter:off
		int[][] trainingSet = {
			{ 1, 1 },
			{ 1, 2 }
		};
		// @formatter:on
    test(predict(trainingSet, new int[] { 1 }), 1);
  }

  @Test
  public void training_F() {
    // @formatter:off
		int[][] trainingSet = {
			{ 1, 1 },
			{ 1, 1 },
			{ 1, 2 },
			{ 1, 2 }
		};
		// @formatter:on
    test(predict(trainingSet, new int[] { 1 }), 1);
  }

  @Test
  public void training_G() {
    // @formatter:off
		int[][] trainingSet = {
			{ 1, 1 },
			{ 1, 1 },
			{ 1, 1 },
			{ 1, 2 }
		};
		// @formatter:on
    test(predict(trainingSet, new int[] { 1 }), 1);
  }

  @Test
  public void training_H() {
    // @formatter:off
		int[][] trainingSet = {
			{ 1, 2, 4 },
			{ 1, 2, 5 },
			{ 1, 2, 6 },
			{ 1, 3, 7 }
		};
		// @formatter:on
    test(predict(trainingSet, new int[] { 1 }), 2);
  }

  @Test
  public void training_I() {
    // @formatter:off
		int[][] trainingSet = {
			{ 1, 1, 1, 1, 1, 1, 1 },
			{ 1, 1, 1, 1, 1, 1, 1 },
			{ 1, 1, 1, 1, 1, 1, 1 },
			{ 1, 1, 1, 1, 1, 1, 2 }
		};
		// @formatter:on
    test(predict(trainingSet, new int[] { 1 }), 1);
  }

  @Test
  public void training_J() {
    // @formatter:off
		int[][] trainingSet = {
			{ 1, 1, 1, 1, 2, 3, 5 },
			{ 1, 1, 1, 1, 2, 3, 6 },
			{ 1, 1, 1, 1, 2, 3, 7 },
			{ 1, 1, 1, 1, 2, 3, 8 }
		};
		// @formatter:on
    test(predict(trainingSet, new int[] { 2 }), 3);
    test(predict(trainingSet, new int[] { 1, 1, 2 }), 3);
  }

  @Test
  public void training_K() {
    // @formatter:off
		int[][] trainingSet = {
			{ 1, 1, 2, 1, 1, 1, 3 },
			{ 1, 1, 1, 1, 1, 1, 1 },
			{ 1, 1, 2, 1, 1, 1, 3 },
			{ 1, 1, 1, 1, 1, 1, 1 }
		};
		// @formatter:on
    test(predict(trainingSet, new int[] { 2, 1, 1, 1 }), 3);
  }
}
