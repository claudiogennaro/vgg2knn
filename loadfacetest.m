dname = dir('./faces');
faceset = dname.folder;
faceData = imageDatastore(faceset,'IncludeSubfolders',true,'LabelSource','foldernames');
faceData.ReadFcn = @(filename)readAndPreprocessImage224x224(filename);
features = activations(vgg2,faceData,'pool5|7x7_s1','MiniBatchSize', 32, 'OutputAs', 'columns');
fn = normc(features);