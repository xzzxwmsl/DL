public class TestMatrix {
    public static void main(String[] args) {
        double[][] arrayA = { { 1, 2, 3 }, { 1, 2, 3 } };
        double[][] arrayB = { { 2, 2 }, { 1, 1 } };
        double[][] arrayC = { { 1, 1 }, { 2, 2 } };

        Matrix matrixB = new Matrix(arrayB);
        Matrix matrixA = new Matrix(arrayA);
        Matrix matrixC = new Matrix(arrayC);

        System.out.println("B*A:");
        matrixB.mulMatrix(matrixA);
        System.out.println("B:");
        System.out.println(matrixB);

        System.out.println("C+B");
        matrixC.addMatrix(matrixB);
        matrixC.showMatrix();
        System.out.println("showB:");
        matrixB.showMatrix();
        System.out.println("_____________");
        matrixB.mulMatrix(matrixA);
        System.out.println("_____________");
        matrixB.showMatrix();
    }

    public static void show(double[][] a) {
        for (double[] i : a) {
            for (double j : i) {
                System.out.print(j + " ");
            }
            System.out.println();
        }
    }
}
