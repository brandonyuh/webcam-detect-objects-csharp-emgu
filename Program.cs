using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.Dnn;

// Read the model and weights From the YOLO Detection Model https://pjreddie.com/darknet/yolo/
// Put https://pjreddie.com/media/files/yolov3.weights in the detection folder! It's too big for GitHub
var net = Emgu.CV.Dnn.DnnInvoke.ReadNetFromDarknet(
    "./detection/yolov3.cfg",
    "./detection/yolov3.weights"
);

// Read the labels for detected objects
var classLabels = File.ReadAllLines("./detection/coco.names");

net.SetPreferableBackend(Emgu.CV.Dnn.Backend.OpenCV);
net.SetPreferableTarget(Emgu.CV.Dnn.Target.Cpu);

// Get the video capture
var videoCapture = new VideoCapture(0, VideoCapture.API.DShow);

Mat frame = new();
VectorOfMat output = new();

VectorOfRect boxes = new();
VectorOfFloat scores = new();
VectorOfInt indices = new();

while (true)
{
    videoCapture.Read(frame);

    //resize the frame to 40% of the original size to save on computation
    CvInvoke.Resize(frame, frame, new System.Drawing.Size(0, 0), .4, .4);

    // Clear the previous detections
    boxes = new();
    indices = new();
    scores = new();

    var image = frame.ToImage<Bgr, byte>();

    //Scale the image down to save on computation, also swap the R and B channels
    var input = DnnInvoke.BlobFromImage(image, 1 / 255.0, swapRB: true);

    // Send the blob to the network
    net.SetInput(input);

    // Get the output from the network
    net.Forward(output, net.UnconnectedOutLayersNames);

    for (int i = 0; i < output.Size; i++)
    {
        var mat = output[i];
        var data = (float[,])mat.GetData();

        for (int j = 0; j < data.GetLength(0); j++)
        {
            float[] row = Enumerable.Range(0, data.GetLength(1)).Select(x => data[j, x]).ToArray();

            var rowScore = row.Skip(5).ToArray();
            var classId = rowScore.ToList().IndexOf(rowScore.Max());
            var confidence = rowScore[classId];

            //if the model thinks it's 80% sure or more that it's found an object
            if (confidence > 0.8f)
            {
                //draw the box around the detected object
                var centerX = (int)(row[0] * frame.Width);
                var centerY = (int)(row[1] * frame.Height);
                var boxWidth = (int)(row[2] * frame.Width);
                var boxHeight = (int)(row[3] * frame.Height);

                var x = (int)(centerX - (boxWidth / 2));
                var y = (int)(centerY - (boxHeight / 2));

                boxes.Push(
                    new System.Drawing.Rectangle[]
                    {
                        new System.Drawing.Rectangle(x, y, boxWidth, boxHeight)
                    }
                );
                indices.Push(new int[] { classId });
                scores.Push(new float[] { confidence });
            }
        }
    }

    var bestDetectionIndices = DnnInvoke.NMSBoxes(boxes.ToArray(), scores.ToArray(), .8f, .8f);

    var frameOut = frame.ToImage<Bgr, byte>();

    for (int i = 0; i < bestDetectionIndices.Length; i++)
    {
        //convert the score to a percentage
        int displayScore = (int) (scores[i]*100);
        int index = bestDetectionIndices[i];
        var box = boxes[index];

        //draw a green rectangle around the detected object
        CvInvoke.Rectangle(frameOut, box, new MCvScalar(0, 255, 0), 2);

        //put the text 20 pixels above the detected object
        CvInvoke.PutText(
            frameOut,
            classLabels[indices[index]] + " " + displayScore + "%",
            new System.Drawing.Point(box.X, box.Y - 20),
            Emgu.CV.CvEnum.FontFace.HersheyPlain,
            1.0,
            new MCvScalar(0, 0, 255),
            2
        );
    }

    //restore the image to its original size
    CvInvoke.Resize(frameOut, frameOut, new System.Drawing.Size(0, 0), 4, 4);
    CvInvoke.Imshow("Webcam Object Detection", frameOut);

    //break on escape key
    if (CvInvoke.WaitKey(1) == 27)
        break;
}
