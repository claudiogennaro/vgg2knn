% face recogntion with VGG face resent50 network and kNN classifier
%
% this code has been developed starting from the example Face Detection and
% "Tracking Using Live Video Acquisition"
% https://it.mathworks.com/help/vision/examples/face-detection-and-tracking-using-live-video-acquisition.html
%
% Load the VGG2 face neural network
% https://github.com/ox-vgg/vgg_face2
load vggface2.mat;

% scan 'faces' folder for sample faces and extract normalized features
loadfacetest;

faceDetector = vision.CascadeObjectDetector();

% Create the webcam object.
cam = webcam();

% Capture one frame to get its size.
videoFrame = snapshot(cam);
frameSize = size(videoFrame);

% Create the video player object. 
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);

% Get the inputsize of the vgg2
inputSize = vgg2.Layers(1).InputSize;

%% Detection and Tracking
% Capture and process video frames from the webcam in a loop to detect and
% track a face. The loop will run for 400 frames or until the video player
% window is closed.

runLoop = true;


while runLoop 
    
    % Get the next frame.
    videoFrame = snapshot(cam);
    videoFrameGray = rgb2gray(videoFrame);
    
    % Detection mode.
    bbox = faceDetector.step(videoFrameGray);

    if ~isempty(bbox)
        
        nboxes = size(bbox,1);

        for i=1:nboxes
            % Find corner points inside the detected region.
            %points = detectMinEigenFeatures(videoFrameGray, 'ROI', bbox(1, :));

            % Convert the rectangle represented as [x, y, w, h] into an
            % M-by-2 matrix of [x,y] coordinates of the four corners. This
            % is needed to be able to transform the bounding box to display
            % the orientation of the face.
            bboxPoints = bbox2points(bbox(i, :));  

            % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4] 
            % format required by insertShape.
            bboxPolygon = reshape(bboxPoints', 1, []);

            % Display a bounding box around the detected face.
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);

            % Extract the face form the bounding box
            IFace = imcrop(videoFrame,bbox(i,:));

            % Resize the face to match the input size of the vgg2
            IFace = imresize(IFace,inputSize(1:2));
         
            % take the current query
            % if you do not have gpus this next step could take time
            q = activations(vgg2,IFace,'pool5|7x7_s1','MiniBatchSize', 32, 'OutputAs', 'columns');
            qn = normc(q);
            
            % run knn classifier
            bestlabel = knn(qn,fn,faceData);
            
            % show label name
            videoFrame = insertText(videoFrame, bboxPoints(1,:),char(bestlabel));
        end
        
    end
    
    % Display the annotated video frame using the video player object.
    step(videoPlayer, videoFrame);
    
    % Check whether the video player window has been closed.
    runLoop = isOpen(videoPlayer);
  
end
        


% Clean up.
clear cam;
release(videoPlayer);
release(faceDetector);
