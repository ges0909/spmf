import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.ArrayList;
import java.util.List;

import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.database.Item;
import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.database.Sequence;

class PredictorTestUtils {

    Sequence seqOf(int id, int[] symbols) {
        Sequence sequence = new Sequence(id);
        for (int s : symbols) {
            sequence.addItem(new Item(Integer.valueOf(s)));
        }
        return sequence;
    }

    Sequence seqOf(int[] symbols) {
        return seqOf(-1, symbols);
    }

    List<Sequence> modelOf(int[]... symbolSeries) {
        int id = 0;
        List<Sequence> model = new ArrayList<>();
        for (int[] symbols : symbolSeries) {
            model.add(seqOf(++id, symbols));
        }
        return model;
    }

    void test(Sequence actual, int... expected) {
        int n = 0;
        assertEquals(expected.length, actual.size(), "seq. lengths: ");
        for (int s : expected) {
            assertEquals(s, actual.getItems().get(n).val.intValue(), "symbol #" + ++n);
        }
    }
}
