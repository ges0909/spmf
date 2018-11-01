import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.database.Sequence;
import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.predictor.CPT.CPT.CPTPredictor;
import org.junit.jupiter.api.Test;

import java.util.Collections;
import java.util.Comparator;
import java.util.Map;

public class CPTTest extends PredictorTestUtils {

    // http://www.philippe-fournier-viger.com/ADMA2013_Compact_Prediction_trees.pdf
    CPTPredictor predictor = new CPTPredictor("CPT",
            "splitLength:6 splitMethod:0 recursiveDividerMin:1 recursiveDividerMax:5");

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
        Sequence targetSeq = predictor.Predict(seqOf(new int[]{1, 4}));
        Float max = Collections.max(predictor.getCountTable().values());
        predictor.getCountTable().entrySet().stream().filter(e -> e.getValue() >= max).forEach(e -> System.out.println(e.getKey() + ": " + e.getValue()));
        System.out.println("--");
        predictor.getCountTable().forEach((k, v) -> System.out.println(k + ": " + v));
        test(targetSeq, 2);
    }

    @Test
    public void withCPTplusTestData() {
        // @formatter:off
		int[][] trainingSet = {
			{ 1, 2, 3, 4, 6    },
			{ 4, 3, 2, 5       },
			{ 5, 1, 4, 3, 2    },
			{ 5, 7, 1, 4, 2, 3 }
		};
		// @formatter:on
        predictor.Train(modelOf(trainingSet));
        test(predictor.Predict(seqOf(new int[]{1, 2})), 3);
    }

    @Test
    public void training_K() {
        // @formatter:off
		int[][] trainingSet = {
			// { 1, 1, 2, 1, 1, 1, 3 },
			// { 1, 1, 1, 1, 1, 1, 1 },
			{ 1, 1, 2, 1, 1, 1, 3 },
			{ 1, 1, 1, 1, 1, 1, 1 }
		};
		// @formatter:on
        predictor.Train(modelOf(trainingSet));
        Sequence targetSeq = predictor.Predict(seqOf(new int[]{2, 1, 1, 1}));
        predictor.getCountTable().forEach((k, v) -> System.out.println(k + ": " + v));
        test(targetSeq, 3 /*1 -> result before bug fix*/);
    }
}
