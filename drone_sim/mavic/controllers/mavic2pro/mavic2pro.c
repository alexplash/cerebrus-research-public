
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cjson/cJSON.h>

#include <webots/robot.h>

#include <webots/camera.h>
#include <webots/compass.h>
#include <webots/gps.h>
#include <webots/gyro.h>
#include <webots/inertial_unit.h>
#include <webots/keyboard.h>
#include <webots/led.h>
#include <webots/motor.h>

#define SIGN(x) ((x) > 0) - ((x) < 0)
#define CLAMP(value, low, high) ((value) < (low) ? (low) : ((value) > (high) ? (high) : (value)))

char *read_file(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        return NULL;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }

    long size = ftell(fp);
    if (size < 0) {
        fclose(fp);
        return NULL;
    }

    rewind(fp);

    char *buffer = malloc((size_t)size + 1);
    if (!buffer) {
        fclose(fp);
        return NULL;
    }

    size_t bytes_read = fread(buffer, 1, (size_t)size, fp);
    if (bytes_read != (size_t)size) {
        free(buffer);
        fclose(fp);
        return NULL;
    }

    buffer[size] = '\0';
    fclose(fp);
    return buffer;
}


void apply_action_bins(
  const int action_bins[3],
  const char *expected_flight_command_value,
  double *pitch_disturbance, // x
  double *yaw_disturbance, // y
  double *target_altitude // z
) {

  for (int i = 0; i < 3; i++) { // iterating over x, yaw, z
    
    // positive action
    if (action_bins[i] == 2) {

      // fly forward
      if (i == 0) {
        printf("%s\n", expected_flight_command_value);
        *pitch_disturbance = -2.0;
        return;
      }

      // turn left
      if (i == 1) {
        printf("%s\n", expected_flight_command_value);
        *yaw_disturbance = 1.3;
        return;
      }

      // ascend
      if (i == 2) {
        printf("%s\n", expected_flight_command_value);
        *target_altitude += 0.005;
        return;
      }
    }

    // negative action
    if (action_bins[i] == 0) {

      // fly backward
      if (i == 0) {
        printf("%s\n", expected_flight_command_value);
        *pitch_disturbance = 2.0;
        return;
      }

      // turn right
      if (i == 1) {
        printf("%s\n", expected_flight_command_value);
        *yaw_disturbance = -1.3;
        return;
      }

      // descend
      if (i == 2) {
        printf("%s\n", expected_flight_command_value);
        *target_altitude -= 0.005;
        return;
      }

    }

  }

  printf("%s\n", expected_flight_command_value);

}

bool validate_actions(
  const int action_bins[3],
  const char *expected_flight_command_value,
  double *start_x,
  double *start_y,
  double *start_z,
  double *start_yaw,
  double *curr_x,
  double *curr_y,
  double *curr_z,
  double *curr_yaw
) {
  const double position_threshold = 0.03;
  const double yaw_threshold = 0;
  const double altitude_threshold = 0.03;

  // if action is hover, we just return true
  if (action_bins[0] == 1 && action_bins[1] == 1 && action_bins[2] == 1) {
    return true;
  }

  for (int i = 0; i < 3; i++) { // iterating over x, yaw, z
    
    // positive action
    if (action_bins[i] == 2) {

      // fly forward
      if (i == 0) {
        printf("validating command: %s\n", expected_flight_command_value);
        double dx = *curr_x - *start_x;
        double dy = *curr_y - *start_y;
        double horizontal_distance = sqrt((dx * dx) + (dy * dy));
        return (horizontal_distance > position_threshold);
      }

      // turn left
      if (i == 1) {
        printf("validating command: %s\n", expected_flight_command_value);
        return (fabs(fabs(*curr_yaw) - fabs(*start_yaw)) >= yaw_threshold);
      }

      // ascend
      if (i == 2) {
        printf("validating command: %s\n", expected_flight_command_value);
        return ((*curr_z - *start_z) > altitude_threshold);
      }
    }

    // negative action
    if (action_bins[i] == 0) {

      // fly backward
      if (i == 0) {
        printf("validating command: %s\n", expected_flight_command_value);
        double dx = *curr_x - *start_x;
        double dy = *curr_y - *start_y;
        double horizontal_distance = sqrt((dx * dx) + (dy * dy));
        return (horizontal_distance > position_threshold);
      }

      // turn right
      if (i == 1) {
        printf("validating command: %s\n", expected_flight_command_value);
        return (fabs(fabs(*curr_yaw) - fabs(*start_yaw)) >= yaw_threshold);
      }

      // descend
      if (i == 2) {
        printf("validating command: %s\n", expected_flight_command_value);
        return ((*curr_z - *start_z) < -altitude_threshold);
      }

    }

  }

  return false;

}


int main(int argc, char **argv) {

  // first we read in the manifest json
  const char *manifest_path = getenv("BLA_SIM_MANIFEST_PATH");
  if (!manifest_path || manifest_path[0] == '\0') {
        fprintf(stderr, "BLA_SIM_MANIFEST_PATH is not set.\n");
        return EXIT_FAILURE;
  }

  char *json_text = read_file(manifest_path);
  if (!json_text) {
        fprintf(stderr, "Could not read manifest file: %s\n", manifest_path);
        return EXIT_FAILURE;
  }

  cJSON *root = cJSON_Parse(json_text);
  if (!root) {
      const char *parse_error = cJSON_GetErrorPtr();
      if (parse_error) {
          fprintf(
              stderr,
              "Invalid JSON in %s near: %s\n",
              manifest_path,
              parse_error
          );
      } else {
          fprintf(stderr, "Invalid JSON in %s\n", manifest_path);
      }
      free(json_text);
      return EXIT_FAILURE;
  }

  printf("Loaded simulation manifest: %s\n", manifest_path);
  free(json_text);
  
  cJSON *scenarios = cJSON_GetObjectItemCaseSensitive(root, "scenarios");
  if (!cJSON_IsArray(scenarios)) {
    fprintf(stderr, "\"scenarios\" is missing or is not an array\n");
    cJSON_Delete(root);
    return EXIT_FAILURE;
  }

  int scenario_count = cJSON_GetArraySize(scenarios);
  printf("scenario_count: %d\n", scenario_count);

  wb_robot_init();
  int timestep = (int)wb_robot_get_basic_time_step();

  // Get and enable devices.
  WbDeviceTag camera = wb_robot_get_device("camera");
  wb_camera_enable(camera, timestep);
  WbDeviceTag front_left_led = wb_robot_get_device("front left led");
  WbDeviceTag front_right_led = wb_robot_get_device("front right led");
  WbDeviceTag imu = wb_robot_get_device("inertial unit");
  wb_inertial_unit_enable(imu, timestep);
  WbDeviceTag gps = wb_robot_get_device("gps");
  wb_gps_enable(gps, timestep);
  WbDeviceTag compass = wb_robot_get_device("compass");
  wb_compass_enable(compass, timestep);
  WbDeviceTag gyro = wb_robot_get_device("gyro");
  wb_gyro_enable(gyro, timestep);
  wb_keyboard_enable(timestep);
  WbDeviceTag camera_roll_motor = wb_robot_get_device("camera roll");
  WbDeviceTag camera_pitch_motor = wb_robot_get_device("camera pitch");
  // WbDeviceTag camera_yaw_motor = wb_robot_get_device("camera yaw");  // Not used in this example.

  // Get propeller motors and set them to velocity mode.
  WbDeviceTag front_left_motor = wb_robot_get_device("front left propeller");
  WbDeviceTag front_right_motor = wb_robot_get_device("front right propeller");
  WbDeviceTag rear_left_motor = wb_robot_get_device("rear left propeller");
  WbDeviceTag rear_right_motor = wb_robot_get_device("rear right propeller");
  WbDeviceTag motors[4] = {front_left_motor, front_right_motor, rear_left_motor, rear_right_motor};
  int m;
  for (m = 0; m < 4; ++m) {
    wb_motor_set_position(motors[m], INFINITY);
    wb_motor_set_velocity(motors[m], 1.0);
  }

  // Display the welcome message.
  printf("Start the drone...\n");

  // Wait one second.
  while (wb_robot_step(timestep) != -1) {
    if (wb_robot_get_time() > 1.0)
      break;
  }

  // Constants, empirically found.
  const double k_vertical_thrust = 68.5;  // with this thrust, the drone lifts.
  const double k_vertical_offset = 0.6;   // Vertical offset where the robot actually targets to stabilize itself.
  const double k_vertical_p = 3.0;        // P constant of the vertical PID.
  const double k_roll_p = 50.0;           // P constant of the roll PID.
  const double k_pitch_p = 30.0;          // P constant of the pitch PID.

  // Variables.
  double target_altitude = 1.0;  // The target altitude. Can be changed by the user.

  int curr_timestep = 0;
  int curr_scenario_idx = 0;
  double start_x;
  double start_y;
  double start_z;
  double start_yaw;

  // Main loop
  while (wb_robot_step(timestep) != -1) {
    const double time = wb_robot_get_time();  // in seconds.

    // Retrieve robot position using the sensors.
    const double roll = wb_inertial_unit_get_roll_pitch_yaw(imu)[0];
    const double pitch = wb_inertial_unit_get_roll_pitch_yaw(imu)[1];
    const double altitude = wb_gps_get_values(gps)[2];
    const double roll_velocity = wb_gyro_get_values(gyro)[0];
    const double pitch_velocity = wb_gyro_get_values(gyro)[1];

    // Blink the front LEDs alternatively with a 1 second rate.
    const bool led_state = ((int)time) % 2;
    wb_led_set(front_left_led, led_state);
    wb_led_set(front_right_led, !led_state);

    // Stabilize the Camera by actuating the camera motors according to the gyro feedback.
    wb_motor_set_position(camera_roll_motor, -0.115 * roll_velocity);
    wb_motor_set_position(camera_pitch_motor, -0.1 * pitch_velocity);

    // Transform the keyboard input to disturbances on the stabilization algorithm.
    double roll_disturbance = 0.0;
    double pitch_disturbance = 0.0;
    double yaw_disturbance = 0.0;
    

    // iterate over scenarios
    // each scenario will take 500 timesteps
    // ----------------------------------------------------------------------------
    if (curr_scenario_idx >= scenario_count) {
        printf("Finished all scenarios.\n");
        break;
    }

    cJSON *scenario = cJSON_GetArrayItem(scenarios, curr_scenario_idx);
    if (!cJSON_IsObject(scenario)) {
      fprintf(stderr, "scenario %d is not a JSON object\n", curr_scenario_idx);
      break;
    }

    int action_bin_values[3];
    cJSON *action_bins = cJSON_GetObjectItemCaseSensitive(scenario, "action_bins");
    if (!cJSON_IsArray(action_bins) || cJSON_GetArraySize(action_bins) != 3) {
        fprintf(stderr, "\"action_bins\" is missing or malformed\n");
        break;
    }

    cJSON *expected_flight_command = cJSON_GetObjectItemCaseSensitive(scenario, "expected_flight_command");
    if (!cJSON_IsString(expected_flight_command) || expected_flight_command->valuestring == NULL) {
      fprintf(stderr, "\"expected_flight_command\" is missing or malformed\n");
      break;
    }
    const char *expected_flight_command_value = expected_flight_command->valuestring;

    for (int i = 0; i < 3; i++) {
      cJSON *item = cJSON_GetArrayItem(action_bins, i);

      if (!cJSON_IsNumber(item)) {
          fprintf(stderr, "action_bins[%d] is not a number\n", i);
          break;
      }

      action_bin_values[i] = item->valueint;
    }

    // establish the starting position and orientation for each new action
    if (curr_timestep % 500 == 0) {
      const double *gps_values = wb_gps_get_values(gps);
      start_x = gps_values[0];
      start_y = gps_values[1];
      start_z = gps_values[2];

      const double *rpy = wb_inertial_unit_get_roll_pitch_yaw(imu);
      start_yaw = rpy[2];
    }

    apply_action_bins(
      action_bin_values,
      expected_flight_command_value,
      &pitch_disturbance,
      &yaw_disturbance,
      &target_altitude
    );

    curr_timestep += 1;
    if (curr_timestep % 500 == 0 && curr_timestep > 0) {

      // validate that the scenario resulted in the desired final position & orientation
      // ---------------------------------------------------------------------------------
      // ---------------------------------------------------------------------------------
      const double *curr_gps_values = wb_gps_get_values(gps);
      double curr_x = curr_gps_values[0];
      double curr_y = curr_gps_values[1];
      double curr_z = curr_gps_values[2];

      const double *curr_rpy = wb_inertial_unit_get_roll_pitch_yaw(imu);
      double curr_yaw = curr_rpy[2];

      const bool validation_results = validate_actions(
        action_bin_values,
        expected_flight_command_value,
        &start_x,
        &start_y,
        &start_z,
        &start_yaw,
        &curr_x,
        &curr_y,
        &curr_z,
        &curr_yaw
      );

      if (!validation_results) {
        printf("failed validation for expected action: %s\n", expected_flight_command_value);
        return EXIT_FAILURE;
      }


      curr_scenario_idx += 1;
    }
    // ----------------------------------------------------------------------------


    // Compute the roll, pitch, yaw and vertical inputs.
    const double roll_input = k_roll_p * CLAMP(roll, -1.0, 1.0) + roll_velocity + roll_disturbance;
    const double pitch_input = k_pitch_p * CLAMP(pitch, -1.0, 1.0) + pitch_velocity + pitch_disturbance;
    const double yaw_input = yaw_disturbance;
    const double clamped_difference_altitude = CLAMP(target_altitude - altitude + k_vertical_offset, -1.0, 1.0);
    const double vertical_input = k_vertical_p * pow(clamped_difference_altitude, 3.0);

    // Actuate the motors taking into consideration all the computed inputs.
    const double front_left_motor_input = k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input;
    const double front_right_motor_input = k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input;
    const double rear_left_motor_input = k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input;
    const double rear_right_motor_input = k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input;
    wb_motor_set_velocity(front_left_motor, front_left_motor_input);
    wb_motor_set_velocity(front_right_motor, -front_right_motor_input);
    wb_motor_set_velocity(rear_left_motor, -rear_left_motor_input);
    wb_motor_set_velocity(rear_right_motor, rear_right_motor_input);
  };

  wb_robot_cleanup();

  return EXIT_SUCCESS;

  cJSON_Delete(root);
}
