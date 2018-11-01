import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.database.Sequence;
import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.predictor.CPT.CPTPlus.CPTPlusPredictor;

public class CPTplusTest extends PredictorTestUtils {

	CPTPlusPredictor predictor = new CPTPlusPredictor("CPT+",
			"CCF:false CBS:false CCFmin:1 CCFmax:6 CCFsup:2 splitMethod:0 splitLength:4 minPredictionRatio:1.0 noiseRatio:1.0");

	@Test
	public void withOriginalData() {
		// @formatter:off
		int[][] trainingSet = {
			{ 1, 2, 3, 4, 6    },
			{ 4, 3, 2, 5       },
			{ 5, 1, 4, 3, 2    },
			{ 5, 7, 1, 4, 2, 3 }
		};
		// @formatter:on
		predictor.Train(modelOf(trainingSet));
		test(predictor.Predict(seqOf(new int[] { 1, 2 })), 3);
	}

	@Test
	@Disabled
	public void withCPTTestData() {
		// @formatter:off
		int[][] trainingSet = {
			{ 1, 2, 3, 4, 6    },
			{ 4, 3, 2, 5       },
			{ 5, 1, 4, 3, 2    },
			{ 5, 7, 1, 4, 2, 3 }
		};
		// @formatter:on
		predictor.Train(modelOf(trainingSet));
		test(predictor.Predict(seqOf(new int[] { 1, 4 })), 2);
	}

	@Test
	public void conference2015() {
		// @formatter:off
		int[][] trainingSet = {
			{ 1, 2, 3    }, // ABC
			{ 1, 2       }, // AB
			{ 1, 2, 4, 3 }, // ABDC
			{ 2, 3       }, // BC
			{ 2, 4, 5    }  // BDE
		};
		// @formatter:on
		predictor.Train(modelOf(trainingSet));
		// AB => C
		Sequence seq = predictor.Predict(seqOf(new int[] { 1, 2 } /* AB */));
		predictor.getCountTable().forEach((k, v) -> System.out.println(k + ": " + v));
		// C 2.1337514
		// D 1.6002535
		// E 1.20005
		test(seq, 3 /* C */);
	}

	@Test
	public void training_A() {
		// @formatter:off
		int[][] trainingSet = {
			{ 1, 1 },
			{ 1, 1 }
		};
		// @formatter:on
		predictor.Train(modelOf(trainingSet));
		test(predictor.Predict(seqOf(new int[] { 1 })), 1);
	}

	@Test
	public void training_B() {
		// @formatter:off
		int[][] trainingSet = {
			{ 1, 2 },
			{ 1, 2 }
		};
		// @formatter:on
		predictor.Train(modelOf(trainingSet));
		test(predictor.Predict(seqOf(new int[] { 1 })), 2);
	}

	@Test
	public void training_C() {
		// @formatter:off
		int[][] trainingSet = {
			{ 1, 1, 1 },
			{ 1, 1, 1 }
		};
		// @formatter:on
		predictor.Train(modelOf(trainingSet));
		test(predictor.Predict(seqOf(new int[] { 1 })), 1);
	}

	@Test
	public void training_D() {
		// @formatter:off
		int[][] trainingSet = {
			{ 1, 1, 1, 2 },
			{ 1, 1, 1, 2 }
		};
		// @formatter:on
		predictor.Train(modelOf(trainingSet));
		test(predictor.Predict(seqOf(new int[] { 1 })), 1);
	}

	@Test
	public void training_E() {
		// @formatter:off
		int[][] trainingSet = {
			{ 1, 1 },
			{ 1, 2 }
		};
		// @formatter:on
		predictor.Train(modelOf(trainingSet));
		test(predictor.Predict(seqOf(new int[] { 1 })), 1);
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
		predictor.Train(modelOf(trainingSet));
		test(predictor.Predict(seqOf(new int[] { 1 })), 1);
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
		predictor.Train(modelOf(trainingSet));
		test(predictor.Predict(seqOf(new int[] { 1 })), 1);
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
		predictor.Train(modelOf(trainingSet));
		test(predictor.Predict(seqOf(new int[] { 1 })), 2);
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
		predictor.Train(modelOf(trainingSet));
		test(predictor.Predict(seqOf(new int[] { 1 })), 1);
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
		predictor.Train(modelOf(trainingSet));
		test(predictor.Predict(seqOf(new int[] { 2 })), 3);
		test(predictor.Predict(seqOf(new int[] { 1, 1, 2 })), 3);
	}

	@Test
	public void training_K() {
		// @formatter:off
		int[][] trainingSet = {
//			{ 1, 1, 2, 1, 1, 1, 3 },
//			{ 1, 1, 1, 1, 1, 1, 1 },
			{ 1, 1, 2, 1, 1, 1, 3 },
			{ 1, 1, 1, 1, 1, 1, 1 }
		};
		// @formatter:on
		CPTPlusPredictor predictor = new CPTPlusPredictor("CPT+",
				"CCF:false CBS:false CCFmin:1 CCFmax:6 CCFsup:2 splitMethod:0 splitLength:4 minPredictionRatio:1.0 noiseRatio:1.0");
		predictor.Train(modelOf(trainingSet));
		Sequence seq = predictor.Predict(seqOf(new int[] { 2, 1, 1, 1 }));
		predictor.getCountTable().forEach((k, v) -> System.out.println(k + ": " + v));
		test(seq, 1);
	}
}
