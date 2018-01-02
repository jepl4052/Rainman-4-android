package com.example.jens.rm.rainman;

import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    TextView tempValue;
    TextView humiValue;
    TextView pressValue;
    TextView windValue;
    TextView predValue;
    MultiLayerNetwork model;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.content_main);

        //get access to editable fields in view
        tempValue = (TextView) findViewById(R.id.tempValue);
        humiValue = (TextView) findViewById(R.id.humiValue);
        pressValue = (TextView) findViewById(R.id.pressValue);
        windValue = (TextView) findViewById(R.id.windValue);
        predValue = (TextView) findViewById(R.id.predValue);

        //load pre-trained neural network model from file
        try {
            model = ModelSerializer.restoreMultiLayerNetwork(getAssets().open("rainmanmodel.zip"));
        } catch (IOException e) {
            e.printStackTrace();
        }

        //click on simulate sensor values in app
        final Button simSensValues = findViewById(R.id.getValuesBttn);
        simSensValues.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {

                simulateSensorValues();

            }
        });

        //click on calculate rain in app
        final Button calcPrediction = findViewById(R.id.calculateBttn);
        calcPrediction.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {

                calculatePredictionOutput();

            }
        });
    }

    //randomize sensor values
    private void simulateSensorValues() {

        tempValue.setText("" + randomizer(300, 1700));
        humiValue.setText("" + randomizer(8300, 9500));
        pressValue.setText("" + randomizer(99000, 101500));
        windValue.setText("" + randomizer(200, 400));
    }

    //randomize between given interval
    private double randomizer(int min, int max) {
        int range = (max - min) + 1;
        return ((int)((Math.random() * range) + min))/100.0;
    }

    //predict rain from previously simulated sensor values
    private void calculatePredictionOutput() {

        double temp = Double.parseDouble("" + tempValue.getText());
        double humi = Double.parseDouble("" + humiValue.getText());
        double press = Double.parseDouble("" + pressValue.getText());
        double wind = Double.parseDouble("" + windValue.getText());
        final INDArray input = Nd4j.create(new double[] { (humi/10), (press/100), temp, wind }, new int[] { 1, 4 });

        INDArray output = model.output(input, false);
        double out = 0.00;

        if(!(output.getDouble(0) < 0)) {
            predValue.setText("" + output);
        } else {
            predValue.setText("" + out);
        }
    }
}