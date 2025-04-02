import numpy as np

FRONT_RIGHT = 0
BACK_RIGHT = 1
BACK_LEFT = 2
FRONT_LEFT = 3

class Baseline:

    def act(self, obs):
        
        # Get Attitude as Angle
        attitude = np.arctan2(obs["att"][-1][0], obs["att"][-1][1])
        angle = np.arctan2(obs["des_dir"][0], obs["des_dir"][1]) + attitude

        contacts = np.any(obs["contacts"], axis=0).flatten()
        contacts_sum = sum(contacts)
        if -np.pi / 4.0 <  angle < np.pi / 4.0:
            goal_dir_quadrant = 1 
        elif -3 * np.pi / 4.0 <  angle < - np.pi / 4.0:
            goal_dir_quadrant = 2 
        elif np.pi / 4.0 <  angle < 3 * np.pi / 4.0:
            goal_dir_quadrant = 4 
        else:
            goal_dir_quadrant = 3

        if contacts_sum == 1:
            sign = 0
            if contacts[FRONT_RIGHT]:
                if goal_dir_quadrant == 1:
                    sign = -1
                elif goal_dir_quadrant == 2:
                    sign = -1
                elif goal_dir_quadrant == 3:
                    sign = 1
                elif goal_dir_quadrant == 4:
                    sign = 1
            elif contacts[FRONT_LEFT]:
                if goal_dir_quadrant == 1:
                    sign = 1
                elif goal_dir_quadrant == 2:
                    sign = -1
                elif goal_dir_quadrant == 3:
                    sign = -1
                elif goal_dir_quadrant == 4:
                    sign = 1
            elif contacts[BACK_RIGHT]:
                if goal_dir_quadrant == 1:
                    sign = -1
                elif goal_dir_quadrant == 2:
                    sign = 1
                elif goal_dir_quadrant == 3:
                    sign = 1
                elif goal_dir_quadrant == 4:
                    sign = -1
            elif contacts[BACK_LEFT]:
                if goal_dir_quadrant == 1:
                    sign = 1
                elif goal_dir_quadrant == 2:
                    sign = 1
                elif goal_dir_quadrant == 3:
                    sign = -1
                elif goal_dir_quadrant == 4:
                    sign = -1

            angle = angle + sign * 0.25 * np.pi

        if contacts_sum >= 2:
            if contacts[FRONT_LEFT] and contacts[FRONT_RIGHT]:
                if goal_dir_quadrant == 1:
                    angle = np.sign(angle) * 0.5 * np.pi
                elif goal_dir_quadrant == 2:
                    angle = -0.5 * np.pi
                elif goal_dir_quadrant == 3:
                    angle = angle
                elif goal_dir_quadrant == 4:
                    angle = 0.5 * np.pi
            elif contacts[FRONT_RIGHT] and contacts[BACK_RIGHT]:
                if goal_dir_quadrant == 1:
                    angle = 0
                elif goal_dir_quadrant == 2:
                    angle = angle  
                elif goal_dir_quadrant == 3:
                    angle = np.pi
                elif goal_dir_quadrant == 4:
                    angle = (angle > 0.5 * np.pi) * np.pi
            elif contacts[BACK_RIGHT] and contacts[BACK_LEFT]:
                if goal_dir_quadrant == 1:
                    angle = angle
                elif goal_dir_quadrant == 2:
                    angle = -0.5 * np.pi
                elif goal_dir_quadrant == 3:
                    angle = np.sign(angle) * 0.5 * np.pi
                elif goal_dir_quadrant == 4:
                    angle = 0.5 * np.pi
            elif contacts[BACK_LEFT] and contacts[FRONT_LEFT]:
                if goal_dir_quadrant == 1:
                    angle = 0
                elif goal_dir_quadrant == 2:
                    angle = (angle < -0.5 * np.pi ) * np.pi
                elif goal_dir_quadrant == 3:
                    angle = np.pi
                elif goal_dir_quadrant == 4:
                    angle = angle
        
        return np.array([(angle) / np.pi, 1.0, 0.0])