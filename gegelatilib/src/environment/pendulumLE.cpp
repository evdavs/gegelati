#define _USE_MATH_DEFINES // To get M_PI
#include <math.h>

#include "environment/pendulumLE.h"

const double PendulumLE::MAX_SPEED = 8.0;
const double PendulumLE::MAX_TORQUE = 2.0;
const double PendulumLE::TIME_DELTA = 0.05;
const double PendulumLE::G = 9.81;
const double PendulumLE::MASS = 1.0;
const double PendulumLE::LENGTH = 1.0;
const double PendulumLE::STABILITY_THRESHOLD = 0.1;


void PendulumLE::setAngle(double newValue)
{
    this->currentState.setDataAt(typeid(double), 0, newValue);
}

void PendulumLE::setVelocity(double newValue)
{
    this->currentState.setDataAt(typeid(double), 1, newValue);
}

double PendulumLE::getAngle() const
{
    return *this->currentState.getDataAt(typeid(const double), 0).getSharedPointer<const double>();
}

double PendulumLE::getVelocity() const
{
    return *this->currentState.getDataAt(typeid(const double), 1).getSharedPointer<const double>();;
}

std::vector<std::reference_wrapper<const Data::DataHandler>> PendulumLE::getDataSources()
{
    auto result = std::vector<std::reference_wrapper<const Data::DataHandler>>();
    result.push_back(this->currentState);
    return result;
}

void PendulumLE::reset(size_t seed, Learn::LearningMode mode, uint16_t iterationNumber, uint64_t generationNumber)
{
    // Create seed from seed and mode
    size_t hash_seed = Data::Hash<size_t>()(seed) ^ Data::Hash<Learn::LearningMode>()(mode);

    // Reset the RNG
    this->rng.setSeed(hash_seed);
    /*   if (resetAngle == 0.0) {
           resetAngle = this->rng.getDouble(-M_PI, M_PI); // Initialize resetAngle if not provided
       }
       if (resetVelocity == 0.0) {
           resetVelocity = this->rng.getDouble(-1.0, 1.0); // Initialize resetAngle if not provided
       }*/

    // Set initial state
    this->setAngle(this->rng.getDouble(-M_PI, M_PI));
    this->setVelocity(this->rng.getDouble(-1.0, 1.0));
    this->nbActionsExecuted = 0;
    this->totalReward = 0.0;
}

void PendulumLE::reset(double initialAngle, double initialVelocity){
    this->setAngle(initialAngle);
    this->setVelocity(initialVelocity);
}

double PendulumLE::getActionFromID(const uint64_t& actionID)
{
    double result = (actionID == 0) ? 0.0 : this->availableActions.at((actionID - 1) % availableActions.size());
    return (actionID <= availableActions.size()) ? result : -result;
}

void PendulumLE::doAction(uint64_t actionID)
{
    // Get the action
    double currentAction = getActionFromID(actionID);
    currentAction *= PendulumLE::MAX_TORQUE;

    // Get current state
    double angle = this->getAngle();
    double velocity = this->getVelocity();

    // Compute current reward
    double angleToUpward = fmod((angle + M_PI), (2.f * M_PI)) - M_PI;
    double reward = -((angleToUpward * angleToUpward) + 0.1f * (velocity * velocity) + 0.001f * (currentAction * currentAction));

    // Store and accumulate reward
    this->rewardHistory[this->nbActionsExecuted % REWARD_HISTORY_SIZE] = reward;
    this->nbActionsExecuted++;
    this->totalReward += reward;

    // Update angular velocity
    velocity = velocity + ((-3.0) * G / (2.0 * LENGTH) * (sin(angle + M_PI)) +
                           (3.f / (MASS * LENGTH * LENGTH)) * currentAction) * TIME_DELTA;
    velocity = std::fmin(std::fmax(velocity, -MAX_SPEED), MAX_SPEED);

    // Update angle
    angle = angle + velocity * TIME_DELTA;

    // Save new pendulum state
    this->setAngle(angle);
    this->setVelocity(velocity);
}

bool PendulumLE::isCopyable() const
{
    return true;
}

Learn::LearningEnvironment* PendulumLE::clone() const
{
    return new PendulumLE(*this);
}

double PendulumLE::getScore() const
{
    if (isTerminal()) {
        // 10/ln(nbActions-MinimumNumberOfActionToConsiderStability)
        // The +2 is added to avoid dividing by ln(1) = 0.
        return 10.0 / std::log(((double)this->nbActionsExecuted - (double)PendulumLE::REWARD_HISTORY_SIZE + 2.0));
    }
    else {
        return this->totalReward / (double)this->nbActionsExecuted;
    }
}

bool PendulumLE::isTerminal() const
{
    bool result = false;

    // Is the history long enough to check stability
    if (this->nbActionsExecuted >= PendulumLE::REWARD_HISTORY_SIZE) {
        // Compute mean reward
        double accumulatedReward = 0.0;
        for (auto idx = 0; idx < PendulumLE::REWARD_HISTORY_SIZE; idx++) {
            accumulatedReward += this->rewardHistory[idx];
        }
        accumulatedReward /= (double)PendulumLE::REWARD_HISTORY_SIZE;

        // Check stability
        result = (fabs(accumulatedReward) < PendulumLE::STABILITY_THRESHOLD);
    }

    // The history is too short, or the pendulum was not stabilized.
    return result;
}
