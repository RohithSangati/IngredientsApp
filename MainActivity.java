package com.example.ingredientsapp;


import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Color;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.widget.ImageView;
import android.net.Uri;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.core.app.ActivityCompat;

import org.tensorflow.lite.Interpreter;

import java.util.ArrayList;
import java.io.BufferedReader;
import java.io.InputStreamReader;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.List;
public class MainActivity extends Activity {
    boolean flag = false;
    byte target[];
    private static final int PICK_IMAGE = 100;
    Uri imageUri;
    ImageView showimage;
    Button save;
    Bitmap capture_photo;
    TextView tv;
    ArrayList<String> list = new ArrayList<String>();
    Interpreter interpreter;
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        String permissions[] = {Manifest.permission.READ_EXTERNAL_STORAGE};
        if(!hasPermissions(this, permissions)){
            ActivityCompat.requestPermissions(this, permissions, 42);
        }
        try {
            interpreter = new Interpreter(loadTensorModelFile(),null);
            System.out.println("=================================interpreter "+interpreter);
        } catch (IOException e) {
            e.printStackTrace();
        }
        initClickListner();
        try {
            readReceipe();
        }catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void initClickListner() {
        tv = (TextView) findViewById(R.id.textView2);
        tv.setMovementMethod(new ScrollingMovementMethod());
        showimage = (ImageView) this.findViewById(R.id.show_image);
        Button picture = (Button) this.findViewById(R.id.upload);
        picture.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(View v) {
                openGallery();
            }
        });
    }
    public void predict(){
        Bitmap bitmap = Bitmap.createScaledBitmap(capture_photo, 64, 64, true);
        int height = bitmap.getHeight();
        int width = bitmap.getWidth();
        float value[][][][] = new float[1][64][64][3];
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                int pixel = bitmap.getPixel(i, j);
                int red = Color.red(pixel);//(pixel >> 16) & 0xff;
                int green = Color.green(pixel);//(pixel >> 8) & 0xff;
                int blue = Color.blue(pixel);//(pixel) & 0xff;
                float red1 = Float.parseFloat(red + "");
                float blue1 = Float.parseFloat(blue + "");
                float green1 = Float.parseFloat(green + "");
                value[0][j][i][0] = (float) (blue1 / 255.0);
                value[0][j][i][1] = (float) (green1 / 255.0);
                value[0][j][i][2] = (float) (red1 / 255.0);
            }
        }
        float outputs[][] = new float[1][1126];
        interpreter.run(value,outputs);
        float out[] = new float[1126];
        for(int i=0;i<outputs.length;i++){
            for(int j=0;j<outputs[i].length;j++) {
                out[j] = outputs[i][j];
            }
        }
        int max_index = 0;
        for (int i = 0; i < out.length; i++) {
            max_index = out[i] > out[max_index] ? i : max_index;
        }
        if (max_index > 0)
            max_index = max_index - 1;
        String receipe = list.get(max_index);
        String arr[] = receipe.split("#");
        tv.setText("");
        StringBuilder sb = new StringBuilder();
        sb.append("Predicted Food : "+arr[1]+"\n\n");
        sb.append("Ingredients : "+arr[2]+"\n\n");
        sb.append("Nutrients : "+arr[3]+"\n\n");
        sb.append("Cooking Details : "+arr[4]+"\n\n");
        tv.setText(sb.toString());
    }
    public void readReceipe() throws IOException {
        //AssetFileDescriptor asset_FileDescriptor = this.getAssets().openFd("model/receipe.txt");
        BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open("model/receipe.txt")));
        String line = null;
        while((line = br.readLine()) != null) {
            line = line.trim();
            if(line.length() > 0) {
                list.add(line);
            }
        }
        br.close();
        tv.setText("Total Receipes found in Dataset : "+list.size());
    }
    private MappedByteBuffer loadTensorModelFile() throws IOException {
        AssetFileDescriptor asset_FileDescriptor = this.getAssets().openFd("model/ingredients_model.tflite");
        FileInputStream file_InputStream = new FileInputStream(asset_FileDescriptor.getFileDescriptor());
        FileChannel file_Channel = file_InputStream.getChannel();
        long start_Offset = asset_FileDescriptor.getStartOffset();
        long length = asset_FileDescriptor.getLength();
        MappedByteBuffer buffer = file_Channel.map(FileChannel.MapMode.READ_ONLY,start_Offset,length);
        file_Channel.close();
        return buffer;
    }

    private void openGallery() {
        Intent gallery = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.INTERNAL_CONTENT_URI);
        startActivityForResult(gallery, PICK_IMAGE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && requestCode == PICK_IMAGE) {
            imageUri = data.getData();
            try {
                capture_photo = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
                showimage.setImageBitmap(capture_photo);
                flag = true;
                predict();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public boolean hasPermissions(Context context, String... permissions) {
        if (context != null && permissions != null) {
            for (String permission : permissions) {
                if (ActivityCompat.checkSelfPermission(context, permission) != PackageManager.PERMISSION_GRANTED) {
                    return false;
                }
            }
        }
        return true;
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }
}