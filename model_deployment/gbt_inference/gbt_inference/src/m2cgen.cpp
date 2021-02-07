
namespace gbt {

double score(double * input) {
    double var0;
    if ((input[2]) >= (0.009422321)) {
        if ((input[3]) >= (0.02359379)) {
            var0 = 69.4;
        } else {
            var0 = 53.069668;
        }
    } else {
        if ((input[2]) >= (-0.02183423)) {
            if ((input[3]) >= (0.02703666)) {
                var0 = 51.088238;
            } else {
                var0 = 38.748753;
            }
        } else {
            if ((input[2]) >= (-0.058479838)) {
                var0 = 33.02927;
            } else {
                var0 = 25.84091;
            }
        }
    }
    double var1;
    if ((input[2]) >= (0.0051110727)) {
        if ((input[2]) >= (0.07301323)) {
            if ((input[3]) >= (-0.04985412)) {
                var1 = 59.377037;
            } else {
                var1 = 13.03955;
            }
        } else {
            if ((input[3]) >= (0.000067507266)) {
                var1 = 43.405666;
            } else {
                var1 = 31.388973;
            }
        }
    } else {
        if ((input[3]) >= (0.039086707)) {
            if ((input[2]) >= (0.00079982437)) {
                var1 = 14.918823;
            } else {
                var1 = 36.185055;
            }
        } else {
            if ((input[2]) >= (-0.020756418)) {
                var1 = 27.732695;
            } else {
                var1 = 21.047476;
            }
        }
    }
    return (var0) + (var1);
}

}