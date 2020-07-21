package com.example.open;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import org.tensorflow.lite.Interpreter;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;

import static com.chaquo.python.Python.*;

public class MainActivity extends AppCompatActivity {
    ImageView fina, org;
    Button button, gal, pre;
    String pyo;
    TextView loc;
    byte[] inputData;
    private static final int Gallery_Request_Code = 123;
    PyObject transfer;
    Python py;
    float maxScore = -Float.MIN_VALUE;
    int maxScoreIdx = -1;
    private int INPUT_SIZE = 224;
    private int PIXEL_SIZE = 3;
    private int IMAGE_MEAN = 0;
    private float IMAGE_STD = 255.0f;
    private float MAX_RESULTS = 3;
    private float THRESHOLD = 0.4f;
    private Interpreter interpreter;
    private String modelPath = "Cancer6.tflite";
    private Button b;
    File imgFile;
    TextView tx;
    ImageView im;
    Bitmap mybit,fib;
    InputStream iStream;
    private AssetManager assetManager;
    private ArrayList<String> labels = new ArrayList<String>() {

        {
            add("Malignant");
            add("Benign");
        }

    };
    float[][] result = new float[1][labels.size()];
    @Override
    public boolean onCreateOptionsMenu(Menu menu)
    {
        MenuInflater menuInflater=getMenuInflater();
        menuInflater.inflate(R.menu.menu_bar,menu);
        return super.onCreateOptionsMenu(menu);
    }
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {

        switch (item.getItemId()) {
            case R.id.clear:
            {
                tx.setText("Result");
                fina.setImageDrawable(getResources().getDrawable(android.R.drawable.gallery_thumb));
                org.setImageDrawable(getResources().getDrawable(android.R.drawable.gallery_thumb));
            }
            default:
                return false;
        }
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        //if(!checkPermissionFromDevice())
        //  requestPermission();
        if (!isStarted()) {
            start(new AndroidPlatform(MainActivity.this));
        }
        tx=findViewById(R.id.textView);
        py = Python.getInstance();
        org = findViewById(R.id.imageView);
        button = findViewById(R.id.button);
        fina = findViewById(R.id.fina);
        //loc=findViewById(R.id.loc);
        gal = findViewById(R.id.button2);
        pre = findViewById(R.id.pre);


        gal.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent();
                intent.setType("image/");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(Intent.createChooser(intent, "Pick an image"), Gallery_Request_Code);

            }
        });


        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //new tranfer().execute();
                new runpy().execute();


                new getImage().execute();


            }

        });

        pre.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                loading_model();
                doInference();
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == Gallery_Request_Code && resultCode == RESULT_OK && data != null) {
            Uri imageData = data.getData();

            org.setImageURI(imageData);
            iStream = null;
            try {
                iStream = getContentResolver().openInputStream(imageData);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
            try {
                inputData = getBytes(iStream);
            } catch (IOException e) {
                e.printStackTrace();
            }

            //text.setText(inputData.toString());


        }
    }

    public byte[] getBytes(InputStream inputStream) throws IOException {
        ByteArrayOutputStream byteBuffer = new ByteArrayOutputStream();
        int bufferSize = 1024;
        byte[] buffer = new byte[bufferSize];

        int len = 0;
        while ((len = inputStream.read(buffer)) != -1) {
            byteBuffer.write(buffer, 0, len);
        }
        return byteBuffer.toByteArray();
    }


    private class runpy extends AsyncTask<Void, Void, Void> {

        @Override
        protected Void doInBackground(Void... voids) {


            PyObject tra = PyObject.fromJava(inputData);
            PyObject pyf = py.getModule("ops");
            final byte[] arr = pyf.callAttr("test", tra).toJava(byte[].class);
            fib = BitmapFactory.decodeByteArray(arr, 0, arr.length);


            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    //loc.setText(arr.toString());
                    fina.setImageBitmap(fib);


                }
            });


            return null;
        }
    }

    private class getImage extends AsyncTask<Void, Void, Void> {

        @Override
        protected Void doInBackground(Void... voids) {
            new Thread(new Runnable() {
                @Override
                public void run() {
                     imgFile = new File(Environment.getExternalStorageState() + "/sample1.jpg");
                    Log.i("in the blocks", "miss");
                    if (imgFile.exists()) {


                        mybit = BitmapFactory.decodeFile(imgFile.getAbsolutePath());


                        fina.setImageBitmap(mybit);


                    }
                }


            });
            return null;

        }
    }

    private MappedByteBuffer loadModelFile(Activity activity, String MODEL_FILE) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer;
        byteBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE);


        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                final int val = intValues[pixel++];

                byteBuffer.putFloat((((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                byteBuffer.putFloat((((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                byteBuffer.putFloat((((val) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);

            }
        }
        return byteBuffer;
    }

    private void loading_model() {
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(5);
        options.setUseNNAPI(true);
        try {
            interpreter = new Interpreter(loadModelFile(MainActivity.this, modelPath), options);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void doInference() {
        //Bitmap bitmap = ((BitmapDrawable) im.getDrawable()).getBitmap();
        //fib=BitmapFactory.decodeStream(iStream);
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(fib, INPUT_SIZE, INPUT_SIZE, false);
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap);

        interpreter.run(byteBuffer, result);

        tx.setText(String.valueOf(result));

        Log.i("length", String.valueOf(result.length));
        float confidence = result[0][0];
        Log.i("resutls",String.valueOf(confidence));
        float confidence1=result[0][1];
        Log.i("results2",String.valueOf(confidence1));
        if (confidence>confidence1)
            tx.setText("Benign");
        else
            tx.setText("Malignant");
    }




}