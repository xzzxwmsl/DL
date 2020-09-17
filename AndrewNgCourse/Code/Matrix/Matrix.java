public class Matrix {
    protected int col;
    protected int row;
    private double[][] matrix;

    Matrix(double[][] a) {
        matrix = a;
        row = a.length;
        col = a[0].length;
    }

    Matrix() {
        matrix = new double[1][1];
        matrix[0][0] = 0;
    }

    public static boolean checkMultiplication(Matrix a, Matrix b) {
        if (a.col == b.row) {
            return true;
        } else {
            return false;
        }
    }

    public static boolean checkAdd(Matrix a, Matrix b) {
        if (a.col == b.col && a.row == b.row) {
            return true;
        } else {
            return false;
        }
    }

    public static double[][] aMulb(Matrix a, Matrix b) throws MatrixError {
        if (checkMultiplication(a, b)) {
            double[][] value = new double[a.row][b.col];
            for (int i = 0; i < a.row; i++) {
                for (int j = 0; j < b.col; j++) {
                    for (int k = 0; k < a.col; k++) {
                        value[i][j] += a.matrix[i][k] * b.matrix[k][j];
                    }
                }
            }
            System.out.println("Test:");
            TestMatrix.show(value);
            return value;
        } else {
            throw new MatrixError("不能相乘");
        }
    }

    public static double[][] aAddb(Matrix a, Matrix b) throws MatrixError {
        if (checkAdd(a, b)) {
            double[][] value = new double[a.row][b.col];
            for (int i = 0; i < a.row; i++) {
                for (int j = 0; j < b.col; j++) {
                    value[i][j] = a.matrix[i][j] + b.matrix[i][j];
                }
            }
            return value;
        } else {
            throw new MatrixError("不能相加");
        }
    }

    public int getRow() {
        return row;
    }

    public int getCol() {
        return col;
    }

    protected void mulMatrix(Matrix m) {
        double[][] temp = new double[this.row][m.col];
        try {
            temp = Matrix.aMulb(this, m);
            matrix = temp.clone();
        } catch (MatrixError e) {
            System.out.println(e);
        }

        reNewField();
    }

    protected void addMatrix(Matrix m) {
        double[][] temp = new double[this.row][m.col];
        try {
            temp = Matrix.aAddb(this, m);
            matrix = temp.clone();
        } catch (MatrixError e) {
            System.out.println(e);
        }

        reNewField();
    }

    public void showMatrix() {
        for (double[] i : matrix) {
            for (double j : i) {
                System.out.print(j + " ");
            }
            System.out.println();
        }
    }

    @Override
    public String toString() {
        String s = "";
        for (double[] i : matrix) {
            for (double j : i) {
                s += String.valueOf(j) + "\t\t\t";
            }
            s += '\n';
        }
        return s;
    }

    private void reNewField() {
        row = matrix.length;
        col = matrix[0].length;
    }

}

class Vector extends Matrix {

}