// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package org.pytorch.demo.objectdetection;

import android.graphics.PointF;
import java.util.ArrayList;

class Result {
    PointF rect;

    public Result( PointF rect) {
        this.rect = rect;
    }
};

public class PrePostProcessor {
    // for yolov5 model, no need to apply MEAN and STD
    public final static float[] NO_MEAN_RGB = new float[] {0.485f, 0.456f, 0.406f};
    public final static float[] NO_STD_RGB = new float[] {0.229f, 0.224f, 0.225f};

    // model input image size
    //public final static int INPUT_WIDTH = 960;
    //public final static int INPUT_HEIGHT = 544;
    public final static int OUTPUT_COLUMN = 2; // x and y



    static ArrayList<Result> outputsToPredictions( int[] outputs, float imgScaleX, float imgScaleY, float ivScaleX, float ivScaleY, float startX, float startY ) {
        ArrayList<Result> results = new ArrayList<>();
        final int outputs_length = outputs.length/OUTPUT_COLUMN ;
        if (outputs_length!=0){
        for (int i = 0; i< outputs_length; i++) {
            float x = outputs[i* OUTPUT_COLUMN+0];
            float y = outputs[i* OUTPUT_COLUMN +1];
            x = (int)(imgScaleX * x);
            y = (int)(imgScaleY * y);

            PointF rect = new PointF((int)(startX+ ivScaleX*x),  (int)(startY+ ivScaleY*y));
            Result result = new Result(rect);
            results.add(result);
        }}
        return results;
    }
}
