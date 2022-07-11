#include <iostream>
#include <string>
#include <sstream>
#include <thread>
#include <chrono>
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <tuple>
#include <stdlib.h> //rsrand, rand
#include <time.h>   //random
#include <iomanip>  // std::setprecision


typedef Eigen::Matrix<char,4,4> MatrixX4c;

using Eigen::MatrixXf;

//-----------------------------------------------------------------

std::vector<std::map<std::string, double>> buildQTable(int &n_states)
{

    std::vector<std::string> action = {"up", "down", "left", "right"};

    std::vector<std::map<std::string, double>> qtable;

    for (int ii = 0; ii < n_states; ii++)
    {

        std::map<std::string, double> qi;
        for (auto &jj : action)
        {
            qi.insert(std::pair<std::string, double>(jj, 0.0));
        }

        qtable.push_back(qi);
    }

    int i = 0;
    for (auto &ii : qtable)
    {

        std::cout << " ------ " << i << " ------ "
                  << "\n";

        for (auto &jj : ii)
        {

            std::cout << jj.first << " : " << jj.second << "\n";
        }
        i++;
    }

    return qtable;
}


//--------------------------------------------------------------------

std::string chooseAction(std::vector<std::map<std::string, double>> &q_table, const int &state, double &epsilon)
{


    double rand_value = ((double)rand() / (RAND_MAX));
;
    std::string action;

    if (rand_value < epsilon)
    {
        std::map<std::string, double> state_action = q_table[state];

        double max_state = 0.0;

        for (auto &ii : state_action)
        {

            if (ii.second >= max_state)
            {

                max_state = ii.second;
                action = ii.first;
            }
        }
    }

    else
    {

        int r = rand() % 4;
      
        switch (r)
        {
        case 0:
            action = "up";
            break;

        case 1:
            action = "down";
            break;

        case 2:
            action = "left";
            break;

        case 3:
            action = "right";
            break;
        }
    }

    return action;
}

//--------------------------------------------------------------------

std::tuple<std::tuple<int, int>, int, bool> getEnvFeedback(std::tuple<int, int> &state, std::string &action, std::tuple<int, int> &TERMINAL, std::vector<std::tuple<int, int>> &HOLE)
{

    int LENGTH = 6;
    std::tuple<int, int> nextState;
    int reward = 0;
    bool end = false;

    int a = std::get<0>(state);
    int b = std::get<1>(state);

    if (action == "up")
    {

        a -= 1;
        if (a < 0)
        {
            a = 0;
        }

        nextState = {a, b};

        if (nextState == TERMINAL)
        {
            reward = 1;
            end = true;
        }
        if (nextState == HOLE[0] || nextState == HOLE[1] || nextState == HOLE[2])
        {
            reward = -1;
            end = true;
        }
    }

    if (action == "down")
    {

        a += 1;
        if (a >= LENGTH)
        {
            a = LENGTH - 1;
        }

        nextState = {a, b};

        if (nextState == TERMINAL)
        {
            reward = 1;
            end = true;
        }
        if (nextState == HOLE[0] || nextState == HOLE[1] || nextState == HOLE[2])
        {
            reward = -1;
            end = true;
        }
    }

    if (action == "left")
    {

        b -= 1;
        if (b < 0)
        {
            b = 0;
        }

        nextState = {a, b};

        if (nextState == TERMINAL)
        {
            reward = 1;
            end = true;
        }
        if (nextState == HOLE[0] || nextState == HOLE[1] || nextState == HOLE[2])
        {
            reward = -1;
            end = true;
        }
    }

    if (action == "right")
    {

        b += 1;
        if (b >= LENGTH)
        {
            b = LENGTH - 1;
        }

        nextState = {a, b};

        if (nextState == TERMINAL)
        {
            reward = 1;
            end = true;
        }
        if (nextState == HOLE[0] || nextState == HOLE[1] || nextState == HOLE[2])
        {
            reward = -1;
            end = true;
        }
    }

    std::tuple<std::tuple<int, int>, int, bool> feedback = {nextState, reward, end};

    return feedback;
}

//--------------------------------------------------------------------

std::vector<std::tuple<int, int>> learn(std::vector<std::map<std::string, double>> &qtable, std::tuple<int,int> START, double &EPSILON, std::tuple<int, int> &TERMINAL, std::vector<std::tuple<int, int>> &HOLE ){
    
    int LENGTH = 6;
    std::vector<std::tuple<int, int>> transitions;
    std::tuple<int,int> state = START;
    bool end  = false;

    while (end == false){

        int a = std::get<0>(state);
        int b = std::get<1>(state);
        std::string action = chooseAction(qtable, a * LENGTH + b, EPSILON);
        std::tuple<std::tuple<int, int>, int, bool> feedback = getEnvFeedback(state, action, TERMINAL, HOLE);
        std::tuple<int, int> nextState = std::get<0>(feedback);
        end = std::get<2>(feedback);
        transitions.push_back(nextState);
        state = nextState;

    }

    return transitions;

}

//--------------------------------------------------------------------

int main()
{

    srand(time(NULL));
    int LENGTH = 6;
    int N_STATES = LENGTH * LENGTH;
    int EPISODES = 4000;
    int MAX_STEPS = 300;
    double EPSILON = 0.5;
    double GAMMA = 0.9;
    double ALPHA = 0.1;
    std::tuple<int, int> TERMINAL = {5, 5};
    std::vector<std::tuple<int, int>> HOLE = {{2, 2}, {0, 2}, {2, 1}};// {{3, 0}, {3, 1}, {3, 3}};
    std::tuple<int, int> START = {0, 0};

    std::vector<std::map<std::string, double>> qtable = buildQTable(N_STATES);
    std::tuple<int, int> state;
    bool end = false;

  
    int episode = 0;

    while ((end != true) || (episode < EPISODES))
    {
        episode++;
        state = START;

        for (int step = 0; step < MAX_STEPS; step++)
        {

            int a = std::get<0>(state);
            int b = std::get<1>(state);

            std::string action = chooseAction(qtable, a * LENGTH + b, EPSILON);
      

            std::tuple<std::tuple<int, int>, int, bool> feedback = getEnvFeedback(state, action, TERMINAL, HOLE);

            std::tuple<int, int> nextState = std::get<0>(feedback);
            int na = std::get<0>(nextState);
            int nb = std::get<1>(nextState);

            int reward = std::get<1>(feedback);
            end = std::get<2>(feedback);
            double q_predict;
            double q_target;

            for (auto &ii : qtable[a * LENGTH + b])
            {

                if (ii.first == action)
                {

                    q_predict = ii.second;
                }
            }



            std::map<std::string, double> nextActions = qtable [na * LENGTH + nb];
            double maxValueNextState = 0;

            for (auto &ii : nextActions){

                if (ii.second>= maxValueNextState){
                    maxValueNextState = ii.second;
                }
            }


            if (nextState != TERMINAL){

                q_target = reward + GAMMA * maxValueNextState;

            }
            else {
                q_target = reward;
            }

            //update
          
            for (auto &ii : qtable [a * LENGTH +b]){

                if (ii.first == action){

                    ii.second += ALPHA * (q_target - q_predict);
                }
            }

            state = nextState;
          

        }


    }

    for (auto &ii :qtable){
        for (auto &jj : ii){
            std::cout<<std::setprecision(10) << jj.second << " : ";
        }
        std::cout <<"\n";
    }

    std::vector<std::tuple<int, int>> transitions = learn(qtable, START, EPSILON, TERMINAL, HOLE );

    for (auto &ii : transitions){

        std::cout << std::get<0>(ii) << " : " << std::get<1>(ii) << "\n";
    }


    
}