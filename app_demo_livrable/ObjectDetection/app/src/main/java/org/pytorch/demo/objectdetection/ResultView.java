// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package org.pytorch.demo.objectdetection;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PointF;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;

import java.util.ArrayList;


public class ResultView extends View {

    private Paint mPaintRectangle;
    private ArrayList<Result> mResults;
    private ArrayList<Result> mResults2;

    public ResultView(Context context) {
        super(context);
    }

    public ResultView(Context context, AttributeSet attrs){
        super(context, attrs);
        mPaintRectangle = new Paint();
        mPaintRectangle.setColor(Color.RED);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        if ((mResults == null)||(mResults.size() ==0)) return;
        for (Result result : mResults) {

            mPaintRectangle.setStrokeWidth(10);
            mPaintRectangle.setStyle(Paint.Style.STROKE);
            PointF rect = new PointF(result.rect.x, result.rect.y);
            // Here is where we draw the detections, you can modify the shape here
            canvas.drawCircle(rect.x, rect.y,20, mPaintRectangle);
            ;


        }
    }


    public void setResults(ArrayList<Result> results) {
        mResults = results;
    }
}
