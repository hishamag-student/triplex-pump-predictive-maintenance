clear; clc; close all;
%% 1. Enhanced Fault Simulation with All Fault Codes
T = 600; % 60 seconds at 10Hz
fs = 10; % Sampling frequency
time = (0:T-1)/fs;
% Base signals
flowData = 1.2 + 0.02*randn(1,T);
pressureData = 1.18 + 0.02*randn(1,T);
% All fault definitions with distinct signatures
faults = [
    struct('range', 50:100,   'code', "0",   'desc', 'Leaking Pump Cylinder', ...
           'flow', @(t) 0.8 + 0.3*sin(2*pi*0.3*t), 'pressure', @(t) -0.4); 
    struct('range', 120:170,  'code', "1",   'desc', 'Abnormal Pressure Oscillation', ...
           'flow', @(t) 0.2*sin(2*pi*0.5*t), 'pressure', @(t) 0.7*sin(2*pi*1.2*t));
    struct('range', 190:240,  'code', "10",  'desc', 'Blocked Pump Inlet', ...
           'flow', @(t) -0.7 + 0.4*sin(2*pi*0.4*t), 'pressure', @(t) 0.5);
    struct('range', 260:310,  'code', "11",  'desc', 'Flow Surge and Inlet Blockage', ...
           'flow', @(t) 0.9*sin(2*pi*0.7*t)-0.6, 'pressure', @(t) 0.6*sin(2*pi*1.5*t));
    struct('range', 330:380,  'code', "100", 'desc', 'Increased Pump Bearing Friction', ...
           'flow', @(t) 0.3*randn(size(t)), 'pressure', @(t) 1.2*randn(size(t)));
    struct('range', 400:450,  'code', "101", 'desc', 'Friction and Oscillation', ...
           'flow', @(t) 0.2*randn(size(t)), 'pressure', @(t) 0.8*sin(2*pi*2*t)+0.5*randn(size(t)));
    struct('range', 470:520,  'code', "111", 'desc', 'Multiple Faults', ...
           'flow', @(t) 0.6*sin(2*pi*1.5*t)+0.7*randn(size(t)), ...
           'pressure', @(t) 0.9*sin(2*pi*1*t)+0.6*randn(size(t)))
];
% Inject all faults
for f = 1:length(faults)
    t = time(faults(f).range) - time(faults(f).range(1));
    flowData(faults(f).range) = flowData(faults(f).range) + faults(f).flow(t);
    pressureData(faults(f).range) = pressureData(faults(f).range) + faults(f).pressure(t);
end
%% 2. Load and Verify Model
modelFile = load('digitalTwinModel.mat');
modelFields = fieldnames(modelFile);
model = modelFile.(modelFields{1});
% Get required variables from model
if isfield(model, 'RequiredVariables')
    requiredVars = model.RequiredVariables;
    numFeaturesExpected = length(requiredVars);
else
    error('Model does not specify RequiredVariables');
end
% Check if model has predictFcn
if ~isfield(model, 'predictFcn')
    error('Model does not contain predictFcn');
end
predictFcn = model.predictFcn;
%% 3. Feature Extraction Function
function features = extractFeatures(flow, pressure, fs, numFeaturesExpected)
    % Initialize all possible features
    allFeatures = struct();
    
    % Basic statistical features
    allFeatures.flowMean = mean(flow);
    allFeatures.flowStd = std(flow);
    allFeatures.flowRMS = rms(flow);
    allFeatures.flowPeak2Peak = max(flow)-min(flow);
    allFeatures.flowSkew = skewness(flow);
    allFeatures.flowKurt = kurtosis(flow);
    
    % Frequency features
    L = length(flow);
    Y = abs(fft(flow-mean(flow)));
    P2 = Y/L;
    P1 = P2(1:floor(L/2)+1);
    [~,idx] = max(P1(2:end));
    allFeatures.flowDominantFreq = (idx(1))*fs/L;
    
    % Pressure features
    allFeatures.pressureMean = mean(pressure);
    allFeatures.pressureStd = std(pressure);
    allFeatures.pressureRMS = rms(pressure);
    allFeatures.pressurePeak2Peak = max(pressure)-min(pressure);
    allFeatures.pressureSkew = skewness(pressure);
    allFeatures.pressureKurt = kurtosis(pressure);
    
    % Combined features
    allFeatures.flowPressureCorr = corr(flow', pressure');
    allFeatures.absDiffMean = mean(abs(flow-pressure));
    allFeatures.rmsRatio = allFeatures.flowRMS/allFeatures.pressureRMS;
    
    % Convert to cell array and select requested number of features
    featureNames = fieldnames(allFeatures);
    if numFeaturesExpected > length(featureNames)
        error('Model expects %d features but only %d are available',...
              numFeaturesExpected, length(featureNames));
    end
    
    % Return only the requested number of features
    features = zeros(1, numFeaturesExpected);
    for i = 1:numFeaturesExpected
        features(i) = allFeatures.(featureNames{i});
    end
end
%% 4. MQTT Setup
serverAddress = "tcp://b37.mqtt.one";
portNumber = 1883;
username = "58citw8880";
password = "590degioqu";
topicName = "58citw8880/";
try
    mqClient = mqttclient(serverAddress, ...
        'Port', portNumber, ...
        'Username', username, ...
        'Password', password);
    disp("Connected to MQTT broker successfully.");
catch ME
    error('Failed to connect to MQTT broker: %s', ME.message);
end
%% 5. Real-Time Detection with Sliding Window
windowSize = 60; % 6-second window
stepSize = 10;   % 1-second step
numWindows = floor((T-windowSize)/stepSize)+1;
% Initialize results
detectedFaults = strings(numWindows,1);
actualFaults = strings(numWindows,1);
confidences = zeros(numWindows,1);
timeStamps = zeros(numWindows,1);
% Create fault code to description mapping
faultDescMap = containers.Map(...
    ["0", "1", "10", "11", "100", "101", "110", "111"], ...
    ["Leaking Pump Cylinder", ...
     "Abnormal Pressure Oscillation", ...
     "Blocked Pump Inlet", ...
     "Flow Surge and Inlet Blockage", ...
     "Increased Pump Bearing Friction", ...
     "Friction and Oscillation", ...
     "Normal Operation", ...
     "Multiple Faults"]);
for i = 1:numWindows
    % Get current window
    idx = (1:windowSize) + (i-1)*stepSize;
    flowWin = flowData(idx);
    pressureWin = pressureData(idx);
    timeStamps(i) = time(idx(1));
    
    % Extract features
    feat = extractFeatures(flowWin, pressureWin, fs, numFeaturesExpected);
    
    % Create feature table with correct variable names
    featureTable = array2table(feat, 'VariableNames', requiredVars);
    
    % Predict fault
    try
        [predictedLabel, scores] = predictFcn(featureTable);
        detectedFaults(i) = string(predictedLabel);
        confidences(i) = max(scores);
    catch ME
        warning('Prediction failed at %.1fs: %s', timeStamps(i), ME.message);
        detectedFaults(i) = "Unknown";
        confidences(i) = 0;
    end
    
    % Determine actual fault
    midPoint = round(mean(idx));
    actualFaults(i) = "110"; % Normal
    for f = 1:length(faults)
        if ismember(midPoint, faults(f).range)
            actualFaults(i) = faults(f).code;
            break;
        end
    end
    
    % Prepare MQTT message
    if isKey(faultDescMap, detectedFaults(i))
        faultDesc = faultDescMap(detectedFaults(i));
    else
        faultDesc = 'Unknown Fault';
    end
    
    msgStruct = struct(...
        'timestamp', datestr(datetime('now')), ...
        'Time', timeStamps(i), ...
        'FaultCode', detectedFaults(i), ...
        'AlertMessage', faultDesc, ...
        'confidence', confidences(i), ...
        'Flow', feat(1), ...
        'Pressure', feat(7), ...
        'flow_std', feat(2), ...
        'pressure_std', feat(8));
    
    jsonMsg = jsonencode(msgStruct);
    
    % Print to console
    fprintf('\n[%.1f s] Detected: %s (Code: %s, Confidence: %.2f)\n', ...
        timeStamps(i), faultDesc, detectedFaults(i), confidences(i));
    fprintf('Flow: %.3f ± %.3f, Pressure: %.3f ± %.3f\n', ...
        feat(1), feat(2), feat(7), feat(8));
    
    % Send to MQTT broker
    try
        write(mqClient, topicName, jsonMsg);
        fprintf('Published to MQTT: %s\n', jsonMsg);
    catch ME
        warning('Failed to publish to MQTT: %s', ME.message);
    end
    
    pause(1); % Simulate real-time processing
end
%% 6. Performance Analysis
accuracy = mean(detectedFaults == actualFaults);
fpr gintf('\nOverall Detection Accuracy: %.2f%%\n', accuracy*100);
% Confusion matrix
figure;
cm = confusionchart(actualFaults, detectedFaults, ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');
title('Fault Detection Performance');
%% 7. Visualization
figure;
subplot(3,1,1);
plot(time, flowData);
title('Flow Data with All Faults');
xlabel('Time (s)');
ylabel('Flow');
grid on;
subplot(3,1,2);
plot(time, pressureData);
title('Pressure Data with All Faults');
xlabel('Time (s)');
ylabel('Pressure');
grid on;
% Prepare fault codes for plotting
allFaultCodes = ["110"; unique([faults.code]')]; % Include normal operation code
subplot(3,1,3);
hold on;
plot(timeStamps, grp2idx(actualFaults), 'r-', 'LineWidth', 2);
stem(timeStamps, grp2idx(detectedFaults), 'b', 'filled');
legend('Actual', 'Detected', 'Location', 'best');
title('Fault Detection Results (All Codes)');
xlabel('Time (s)');
ylabel('Fault Code');
yticks(1:length(allFaultCodes));
yticklabels(allFaultCodes);
grid on;
ylim([0.5 length(allFaultCodes)+0.5]);
%% Clean up
clear mqClient;
disp('Monitoring completed and MQTT connection closed.');
