package org.pytorch.demo.objectdetection;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.SystemClock;
import android.util.Log;
import android.view.TextureView;
import android.view.ViewStub;
import android.widget.ImageView;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageProxy;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.PyTorchAndroid;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;

public class ObjectDetectionActivity extends AbstractCameraXActivity<ObjectDetectionActivity.AnalysisResult> {
    private Module mModule = null;
    private ResultView mResultView;
    private ImageView mImageView;

    static class AnalysisResult {
        private final ArrayList<Result> mResults;

        public AnalysisResult(ArrayList<Result> results) {
            mResults = results;
        }
    }

    @Override
    protected int getContentViewLayoutId() {
        return R.layout.activity_object_detection;
    }

    @Override
    protected TextureView getCameraPreviewTextureView() {
        mResultView = findViewById(R.id.resultView);
        return ((ViewStub) findViewById(R.id.object_detection_texture_view_stub))
                .inflate()
                .findViewById(R.id.object_detection_texture_view);
    }

    @Override
    protected void applyToUiAnalyzeImageResult(AnalysisResult result) {
        mResultView.setResults(result.mResults);
        mResultView.invalidate();
    }

    // Equivalent of the Python's gt, example : gt(float[0,1.4,5.9,3], 4.1) -> float[0,0,5.9,0]
    public float[] gt(float[] output,float thres){
        float[] rep = new float[output.length];
        for (int i = 0; i < output.length; i++){
            if (output[i] > thres ){
                rep[i]=output[i];
            }

        }
        return rep;}



    // Return an int list of the coordinate of the non null values of the flatten heatmap return by the model ([x1,y1,x2,y2,...])
    public int[] nonzero(int format ,float[] output){
        ArrayList<Integer> rep= new ArrayList<>();
        for (int i = 0 ; i <output.length ; i++){
            if (output[i] != 0){
                rep.add(i%format);
                rep.add(i / format);
            }
        }
        return rep.stream().mapToInt(Integer::intValue).toArray();
    }


    private Bitmap imgToBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[2].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[0].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }


    @Override
    @WorkerThread
    @Nullable
    protected AnalysisResult analyzeImage(ImageProxy image, int rotationDegrees) {
        if (mModule == null) {
            mModule = PyTorchAndroid.loadModuleFromAsset(getAssets(), "S_MobileNet_noQ.pt");
        }

        Bitmap resizedBitmap = imgToBitmap(image.getImage());

        ImageView mImageView = findViewById(R.id.imageView);
        mImageView.setImageBitmap(resizedBitmap);


        float mImgScaleX = (float) resizedBitmap.getWidth() / (resizedBitmap.getWidth()/ 4);
        float mImgScaleY = (float) resizedBitmap.getHeight() / (resizedBitmap.getHeight()/ 4);

        System.out.println("Scale on X  :  "+mImgScaleX);
        System.out.println("Scale on Y  :  "+mImgScaleY);

        float mIvScaleX = (resizedBitmap.getWidth() > resizedBitmap.getHeight() ? (float) mImageView.getWidth() / resizedBitmap.getWidth() : (float) mImageView.getHeight() / resizedBitmap.getHeight());
        float mIvScaleY = (resizedBitmap.getHeight() > resizedBitmap.getWidth() ? (float) mImageView.getHeight() / resizedBitmap.getHeight() : (float) mImageView.getWidth() / resizedBitmap.getWidth());

        float mStartX = (mImageView.getWidth() - mIvScaleX * resizedBitmap.getWidth()) / 2;
        float mStartY = (mImageView.getHeight() - mIvScaleY * resizedBitmap.getHeight()) / 2;

        final long startTime = SystemClock.elapsedRealtime();

        Matrix matrix = new Matrix();
        matrix.postRotate(90.0f);

        final long startTime3 = SystemClock.elapsedRealtime();
        final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(3 * resizedBitmap.getWidth() * resizedBitmap.getHeight());
        final long inferenceTime3 = SystemClock.elapsedRealtime() - startTime3;
        Log.d("Demo app", "inference time to allocate Float Buffer (ms): " + inferenceTime3);

        final long startTime4 = SystemClock.elapsedRealtime();
        TensorImageUtils.bitmapToFloatBuffer(resizedBitmap, 0, 0, resizedBitmap.getWidth(), resizedBitmap.getHeight(), PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB, floatBuffer, 0);
        final long inferenceTime4 = SystemClock.elapsedRealtime() - startTime4;
        Log.d("Demo app", "inference time to preprocess the mean and the std of the bitmap (ms): " + inferenceTime4);

        final long startTime5 = SystemClock.elapsedRealtime();
        final Tensor inputTensor = Tensor.fromBlob(floatBuffer, new long[]{1,3, resizedBitmap.getHeight(), resizedBitmap.getWidth()});
        final long inferenceTime5 = SystemClock.elapsedRealtime() - startTime5;
        Log.d("Demo app", "inference time to create the input Tensor (ms): " + inferenceTime5);
        System.out.println("Size of the flatten input Tensor :"+ inputTensor.getDataAsFloatArray().length);

        final long startTime6 = SystemClock.elapsedRealtime();
        final float[] heatmap = mModule.forward(IValue.from(inputTensor)).toTensor().getDataAsFloatArray();
        final long inferenceTime6 = SystemClock.elapsedRealtime() - startTime6;
        System.out.println("Size of the heatmap returned : " +  heatmap.length);
        Log.d("Demo app", "inference time of the forward of the tensor in the model (ms): " + inferenceTime6);

        final long startTime7 = SystemClock.elapsedRealtime();
        float[] heatmap_post_gt = gt(heatmap,(float)0.50) ;
        final long inferenceTime7 = SystemClock.elapsedRealtime() - startTime7;
        Log.d("Demo app", "inference time of gt (ms): " + inferenceTime7);

        final long startTime8 = SystemClock.elapsedRealtime();
        int[] post_processed_heatmap = nonzero(resizedBitmap.getWidth()/4+2,heatmap_post_gt);
        final long inferenceTime8 = SystemClock.elapsedRealtime() - startTime8;
        Log.d("Demo app", "inference time of nonzero (ms): " + inferenceTime8);
        System.out.println(Arrays.toString(post_processed_heatmap));

        if (post_processed_heatmap.length != 0){
            final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
            Log.d("Demo app", "inference time total (ms): " + inferenceTime);

            Log.d("Demo app", "Size of the post processed heatmap : " + post_processed_heatmap.length);

            final long startTime9 = SystemClock.elapsedRealtime();
            final ArrayList<Result> results = PrePostProcessor.outputsToPredictions( post_processed_heatmap, mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY);
            final long inferenceTime9 = SystemClock.elapsedRealtime() - startTime9;
            Log.d("Demo app", "inference time of the PrePostProcessor (ms): " + inferenceTime9);
            return new AnalysisResult(results);
        }
        return null;
    }}

