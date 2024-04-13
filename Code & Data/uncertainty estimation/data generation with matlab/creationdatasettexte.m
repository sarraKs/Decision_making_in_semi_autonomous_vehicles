% Open the output file for writing
fid = fopen('output.txt', 'w');
% Loop over each struct in DatasetRadar1
for i = 1:numel(DatasetRadar1)
    time = DatasetRadar1(i).Time; % Get the time value for this struct
    % Loop over each struct in ActorPoses for this time
    for j = 1:numel(DatasetRadar1(i).ActorPoses)
        % Extract the position and velocity arrays
        pos = DatasetRadar1(i).ActorPoses(j).Position;
        vel = DatasetRadar1(i).ActorPoses(j).Velocity;
        % Compute range, velocity, and AoA
        range = sqrt(pos(1)^2 + pos(2)^2);
        vel_norm = sqrt(vel(1)^2 + vel(2)^2);
        velocity = vel_norm * cos(asin(pos(1) / range));
        AoA = atan2(pos(2), pos(1)); 
        % Write the values to the output file
        fprintf(fid, '%f, %f, %f, %f\n', time, range, velocity, AoA);
    end
end
pos1 = DatasetRadar1(1).ActorPoses(1).Position
vel1 = DatasetRadar1(1).ActorPoses(1).Velocity
% Close the output file
fclose(fid);
