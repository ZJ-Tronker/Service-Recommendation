package min.util;
import org.junit.Test;

public class SimilarityMeasure
{
	@Test
	public void test()
	{
//		double[] v1 = new double[]{5, 5};
//		double[] v2 = new double[]{4.5, 3};
////		double[] v3 = new double[]{0.31, 0.14, 0.27};
////		System.out.println(SimilarityMeasure.JSDivergence(v1,v3));
////		System.out.println(SimilarityMeasure.KLDivergence(v1, average(v1, v2)));
////		System.out.println(SimilarityMeasure.KLDivergence(v2, average(v1, v2)));
//		System.out.println(SimilarityMeasure.cosine(v1, v2));
//		
//		double a = 1/(0.9315776194850602+0.9922778767136675+1+0.98058067569092);
//		System.out.println(a);
//		double b = 0.9315776194850602 * 3 +0.9922778767136675 * 4 + 1*4.5 + 0.98058067569092 * 3;
//		System.out.println(a*b);
		
//		double[] v1 = new double[]{5, 1, 1, 5};
//		double[] v2 = new double[]{3, 4, 4.5, 3};
//		System.out.println(SimilarityMeasure.pearson(v1, v2));
		double k = 3;
//		double result = 3.75 + ((-0.8660254037844388)*(4-2.875) + 0.8660254037844387 * (1-3) + (-0.8660254037844387) * (4.5-3.625))/k;
		double result = 3.625 + ((-0.8660254037844387)*(4.5-3.75) + (0.9622504486493763) * (3-3.625))/k;
		System.out.println(result);
	}
	
	/**
	 * The JS divergence similarity
	 * @param v1
	 * @param v2
	 * @return
	 */
	public static double JSDivergence(double[] v1, double[] v2)
	{
		double result = 0;
		result = 0.5 * (KLDivergence(v1, average(v1, v2)) + 
				KLDivergence(v2, average(v1, v2)));
		return result;
	}
	/**
	 * The JS divergence similarity
	 * @param v1
	 * @param v2
	 * @return
	 */
	public static double KLDivergence(double[] v1, double[] v2)
	{
		double result = 0;
		for(int i = 0; i < v1.length && i < v2.length; i++)
		{
			result += v1[i] * Math.log(v1[i] / v2[i]);
		}
		return result;
	}
	private static double[] average(double[] v1, double[] v2)
	{
		double[] result = new double[v1.length];
		for(int i = 0; i < result.length; i++)
		{
			result[i] = (v1[i] + v2[i]) / 2;
		}
		return result;
	}
	
	public static double pearson(double[] v1, double[] v2)
	{
		double result = 0;
		double v1_ave = average(v1);
		double v2_ave = average(v2);
		for(int i = 0; i < v1.length; i++)
		{
			v1[i] -= v1_ave;
			v2[i] -= v2_ave;
		}
		result = cosine(v1, v2);
		return result;
	}
	
	public static double average(double[] v)
	{
		double result = 0;
		for(double i : v) result += i;
		return result / v.length;
	}
	
	
	/**
	 * The normal cosine similarity
	 * @param v1
	 * @param v2
	 * @return
	 */
	public static double cosine(double[] v1, double[] v2)
	{
		double result = 0;
		result = dotProduct(v1, v2) / (moudle(v1) * moudle(v2));
		return result;
	}
	// compute the moudle of a vector
	private static double moudle(double[] v)
	{
		double result = 0;
		for(int i = 0; i < v.length; i++)
		{
			result += v[i] * v[i];
		}
		return Math.sqrt(result);
	}
	// dot product of two vectors
	private static double dotProduct(double[] v1, double[] v2)
	{
		double result = 0;
		for(int i = 0; i < v1.length && i < v2.length; i++)
		{
			result += v1[i] * v2[i];
		}
		return result;
	}
}
