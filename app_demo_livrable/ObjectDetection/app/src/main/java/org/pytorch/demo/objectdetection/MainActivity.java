// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package org.pytorch.demo.objectdetection;




import android.Manifest;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Color ;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.PyTorchAndroid;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;


import java.io.IOException;
import java.io.File;
import java.io.FileWriter;
import java.lang.Math;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;


public class MainActivity extends AppCompatActivity implements Runnable {

    static {
        if (!NativeLoader.isInitialized()) {
            NativeLoader.init(new SystemDelegate());
        }
        NativeLoader.loadLibrary("pytorch_jni");
        NativeLoader.loadLibrary("torchvision_ops");
    }

    private int mImageIndex = 0;
    // Put the name of the photos that are in the "src/main/assets/assets" folder
    private String[] mTestImages = {"test2.jpeg", "test1.jpeg", "image_3.jpeg","image_4.jpeg"};
    private ImageView mImageView;
    private ResultView mResultView;
    private Button mButtonDetect;
    private ProgressBar mProgressBar;
    private Bitmap mBitmap = null;
    // If you want to test a batch of photos, put the size of the batch here (here it is 50 for example)
    private Bitmap[] mBitmapList = new Bitmap[50];
    private Module mModule = null;
    private float mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY;


    // Open the photo file (file is a string) and return a Bitmap
    protected Bitmap createBitmap(String file){
        try{
        BitmapFactory.Options bit = new BitmapFactory.Options();
        bit.inPreferredConfig = Bitmap.Config.RGBA_F16;
        mBitmap = BitmapFactory.decodeStream(getAssets().open(file),null,bit);}
        catch (IOException e) {
            Log.e("Object Detection", "Error reading assets", e);
            finish();
        }
        return mBitmap;
    }

    // What is done when you open the app
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 1);
        }

        setContentView(R.layout.activity_main);

        // Load the first image of mTestImages
        mBitmap= createBitmap(mTestImages[mImageIndex]);
        System.out.println(mBitmap.getConfig());
        System.out.println("Bitmap Width : " + mBitmap.getWidth());
        System.out.println("Bitmap Height : " + mBitmap.getHeight());

        // Print it
        mImageView = findViewById(R.id.imageView);
        mImageView.setImageBitmap(mBitmap);
        mResultView = findViewById(R.id.resultView);
        mResultView.setVisibility(View.INVISIBLE);

        // When you click on the test button, it changes the image printed
        final Button buttonTest = findViewById(R.id.testButton);
        buttonTest.setText(("Test Image 1/"+mTestImages.length));
        buttonTest.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mResultView.setVisibility(View.INVISIBLE);
                mImageIndex = (mImageIndex + 1) % mTestImages.length;
                buttonTest.setText(String.format("Text Image %d/%d", mImageIndex + 1, mTestImages.length));

                mBitmap= createBitmap(mTestImages[mImageIndex]);
                System.out.println("Bitmap Width : " + mBitmap.getWidth());
                System.out.println("Bitmap Height : " + mBitmap.getHeight());
                mImageView.setImageBitmap(mBitmap);

                }
            }
        );


        final Button buttonSelect = findViewById(R.id.selectButton);
        buttonSelect.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mResultView.setVisibility(View.INVISIBLE);

                final CharSequence[] options = {"Choose from Photos", "Take Picture", "Cancel"};
                AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
                builder.setTitle("New Test Image");

                builder.setItems(options, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int item) {
                        if (options[item].equals("Take Picture")) {
                            Intent takePicture = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                            startActivityForResult(takePicture, 0);
                        } else if (options[item].equals("Choose from Photos")) {
                            Intent pickPhoto = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.INTERNAL_CONTENT_URI);
                            startActivityForResult(pickPhoto, 1);
                        } else if (options[item].equals("Cancel")) {
                            dialog.dismiss();
                        }
                    }
                });
                builder.show();
            }
        });

        final Button buttonLive = findViewById(R.id.liveButton);
        buttonLive.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                final Intent intent = new Intent(MainActivity.this, ObjectDetectionActivity.class);
                startActivity(intent);
            }
        });

        mButtonDetect = findViewById(R.id.detectButton);
        mProgressBar = (ProgressBar) findViewById(R.id.progressBar);
        mButtonDetect.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mButtonDetect.setEnabled(false);
                mProgressBar.setVisibility(ProgressBar.VISIBLE);
                mButtonDetect.setText(getString(R.string.run_model));

                mImgScaleX = (float) mBitmap.getWidth() / ((float)Math.ceil(mBitmap.getWidth()/32.0)*8) ;
                mImgScaleY = (float) mBitmap.getHeight() /((float)Math.ceil(mBitmap.getHeight()/32.0)*8);


                mIvScaleX = (mBitmap.getWidth() > mBitmap.getHeight() ? (float) mImageView.getWidth() / mBitmap.getWidth() : (float) mImageView.getHeight() / mBitmap.getHeight());
                mIvScaleY = (mBitmap.getHeight() > mBitmap.getWidth() ? (float) mImageView.getHeight() / mBitmap.getHeight() : (float) mImageView.getWidth() / mBitmap.getWidth());

                mStartX = (mImageView.getWidth() - mIvScaleX * mBitmap.getWidth()) / 2;
                mStartY = (mImageView.getHeight() - mIvScaleY * mBitmap.getHeight()) / 2;

                Thread thread = new Thread(MainActivity.this);
                thread.start();
            }
        });

            System.out.println("Loading .........");
            mModule = PyTorchAndroid.loadModuleFromAsset(getAssets(), "S_MobileNet_noQ.pt");
            System.out.println("Loading SUCCESS");
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_CANCELED) {
            switch (requestCode) {
                case 0:
                    if (resultCode == RESULT_OK && data != null) {
                        mBitmap = (Bitmap) data.getExtras().get("data");
                        Matrix matrix = new Matrix();
                        matrix.postRotate(0.0f);
                        mBitmap = Bitmap.createBitmap(mBitmap, 0, 0, mBitmap.getWidth(), mBitmap.getHeight(), matrix, true);
                        mImageView.setImageBitmap(mBitmap);
                    }
                    break;
                case 1:
                    if (resultCode == RESULT_OK && data != null) {
                        Uri selectedImage = data.getData();
                        String[] filePathColumn = {MediaStore.Images.Media.DATA};
                        if (selectedImage != null) {
                            Cursor cursor = getContentResolver().query(selectedImage,
                                    filePathColumn, null, null, null);
                            if (cursor != null) {
                                cursor.moveToFirst();
                                int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                                String picturePath = cursor.getString(columnIndex);
                                mBitmap = BitmapFactory.decodeFile(picturePath);
                                Matrix matrix = new Matrix();
                                matrix.postRotate(0.0f);
                                mBitmap = Bitmap.createBitmap(mBitmap, 0, 0, mBitmap.getWidth(), mBitmap.getHeight(), matrix, true);
                                mImageView.setImageBitmap(mBitmap);
                                cursor.close();
                            }
                        }
                    }
                    break;
            }
        }
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



    public Bitmap RGBToBGR( Bitmap bitmap){
        int[] pixels = new int[bitmap.getWidth() * bitmap.getHeight()];
        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int len = bitmap.getWidth() * bitmap.getHeight();

        int[] finalArray = new int[bitmap.getWidth() * bitmap.getHeight()];

        for(int i = 0; i < len; i++) {

            int red = Color.red(pixels[i]);
            int green = Color.green(pixels[i]);
            int blue = Color.blue(pixels[i]);
            finalArray[i] = Color.rgb(blue, green, red);//invert sequence here.
        }
        Bitmap bitmapBGR = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), Bitmap.Config.RGBA_F16);

        bitmapBGR.setPixels(finalArray, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        return bitmapBGR;
    }

    // Write a json file where keys are the names of the files and the values the heatmaps, you have to put the images in 'assets' in a folder here named
    // 'batch_0.nosync' and the images have to be named like 'image_i.jpeg'. You can modify the path in testBitmapList too.
    public void testBitmapList(Bitmap[] mBitmapList,String json_name){
        try{
        BitmapFactory.Options bit = new BitmapFactory.Options();
        bit.inPreferredConfig = Bitmap.Config.RGBA_F16;
        for(int i = 0; i< mBitmapList.length ; i++ ){
            mBitmapList[i]=BitmapFactory.decodeStream(getAssets().open("batch_0.nosync/image_"+i+".jpeg"),null,bit);
        }}
        catch (IOException e) {
            System.out.println("Erreur: impossible d'ouvrir batch_0.nosync/image_i.jpeg");
        }
        ArrayList<float[]> reponse = new ArrayList<float[]>();
        String rep =  "{" ;
        float mean_inference_time = 0 ;
        for (int i =0; i < mBitmapList.length; i++){
            final long startTime = SystemClock.elapsedRealtime();
            Bitmap bitmapBGR =  RGBToBGR(mBitmapList[i]);
            FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(3 * bitmapBGR.getWidth() * bitmapBGR.getHeight());
            TensorImageUtils.bitmapToFloatBuffer(mBitmapList[i], 0, 0, bitmapBGR.getWidth(), bitmapBGR.getHeight(), PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB, floatBuffer, 0);
            Tensor inputTensor = Tensor.fromBlob(floatBuffer, new long[]{1,3, bitmapBGR.getHeight(), bitmapBGR.getWidth()});
            float[] heatmap = mModule.forward(IValue.from(inputTensor)).toTensor().getDataAsFloatArray();
            reponse.add(heatmap);
            if (i != 0){
            rep+= ",\"image_"+i+".jpeg\"" +" : " + Arrays.toString(heatmap) ;
        }
            else{
                rep+= "\"image_"+i+".jpeg\"" +" : " + Arrays.toString(heatmap) ;
            }
            final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
            mean_inference_time += (float)inferenceTime/mBitmapList.length ;
            System.out.println("Testing the image dataset :  "+(float)i/(float)mBitmapList.length * (float)100 + " %");}
        WriteTheJson(rep+"}",json_name);
        Log.d("Demo app", "Mean inference time (ms): " + (int)mean_inference_time);
    }

    ///package hacktrack.createfile;



        public void WriteTheJson(String jsonContents, String json_name) {
            File file = new File(MainActivity.this.getExternalFilesDir(null), json_name);
            System.out.println(MainActivity.this.getExternalFilesDir(null));
            try {
                if (!file.exists() )
                    file.createNewFile();
                FileWriter writer = new FileWriter(file);
                writer.write(jsonContents);
                writer.flush();
                writer.close();
                System.out.println("Correctly written the json file at "+ MainActivity.this.getExternalFilesDir(null)+json_name);
            } catch (IOException e) {
                System.out.println("Erreur: impossible de créer le fichier '"
                        + json_name + "'");
            }
        }



    @Override
    public void run() {
        //Comment the line bellow if you have nothing to test
        //testBitmapList( mBitmapList, "test_heatmaps_T_MobileNet_SO.json");


        final long startTime = SystemClock.elapsedRealtime();

        final long startTime2 = SystemClock.elapsedRealtime();
        final Bitmap resizedBitmap = RGBToBGR(mBitmap) ;
        final long inferenceTime2 = SystemClock.elapsedRealtime() - startTime2;
        Log.d("Demo app", "inference time to convert the Bitmap to BGR (ms): " + inferenceTime2);

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
        float[] heatmap_post_gt = gt(heatmap,(float)0.05) ;
        final long inferenceTime7 = SystemClock.elapsedRealtime() - startTime7;
        Log.d("Demo app", "inference time of gt (ms): " + inferenceTime7);

        final long startTime8 = SystemClock.elapsedRealtime();
        int[] post_processed_heatmap = nonzero((int)Math.ceil(mBitmap.getWidth()/32.0)*8,heatmap_post_gt);
        final long inferenceTime8 = SystemClock.elapsedRealtime() - startTime8;
        Log.d("Demo app", "inference time of nonzero (ms): " + inferenceTime8);
        System.out.println(Arrays.toString(post_processed_heatmap));

        final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
        Log.d("Demo app", "inference time total (ms): " + inferenceTime);

        Log.d("Demo app", "Size of the post processed heatmap : " + post_processed_heatmap.length);

        final long startTime9 = SystemClock.elapsedRealtime();
        final ArrayList<Result> results = PrePostProcessor.outputsToPredictions( post_processed_heatmap, mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY);
        final long inferenceTime9 = SystemClock.elapsedRealtime() - startTime9;
            Log.d("Demo app", "inference time of the PrePostProcessor (ms): " + inferenceTime9);

            runOnUiThread(() -> {
                mButtonDetect.setEnabled(true);
                mButtonDetect.setText(getString(R.string.detect));
                mProgressBar.setVisibility(ProgressBar.INVISIBLE);
                mResultView.setResults(results);
                mResultView.invalidate();
                mResultView.setVisibility(View.VISIBLE);
            });
        }
    }


