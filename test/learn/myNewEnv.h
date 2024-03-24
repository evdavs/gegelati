#ifndef FAKE_FEDERATED_LEARNING_ENVIRONMENT_H
#define FAKE_FEDERATED_LEARNING_ENVIRONMENT_H

#include "learn/learningEnvironment.h"
#include "data/arrayWrapper.h"
#include <array>

namespace Learn {

    /**
     * \brief FakeLearningEnvironment class for testing purposes.
     *
     * This class simulates a fake learning environment for testing and debugging.
     * It returns fake data in the form of an ArrayWrapper with pre-known numbers.
     */
    class myNewEnv : public LearningEnvironment {
      private:
        /// Vector of references to data sources
        std::vector<std::reference_wrapper<const Data::DataHandler>> dataSources;

      public:
        /**
         * \brief Constructor for myNewEnv.
         *
         * \param[in] nbActions Number of actions available for interacting with this Environement.
         * \param[in] fakeData Pre-known data source.
         */
        myNewEnv(uint64_t nbActions, const std::vector<double>& fakeData)
            : LearningEnvironment(nbActions) {
            // Create ArrayWrapper instances for pre-known data sources
            auto* array = new Data::ArrayWrapper<double>(fakeData.size());
            array->setPointer(new std::vector<double>(fakeData));
            dataSources.push_back(*array);
        }

        /**
         * \brief Implementation of the reset method for myNewEnv.
         *
         * \param[in] seed The integer value for controlling the randomness of the myNewEnv.
         * \param[in] mode LearningMode in which the myNewEnv should be reset.
         * \param[in] iterationNumber The integer value to indicate the current iteration number.
         * \param[in] generationNumber The integer value to indicate the current generation number.
         */
        void reset(size_t seed = 0, LearningMode mode = LearningMode::TRAINING, uint16_t iterationNumber = 0, uint64_t generationNumber = 0) override {
            // Reset the fake data (not relevant for this fake environment).
        }

        /**
         * \brief Implementation of the doAction method for myNewEnv.
         *
         * \param[in] actionID The integer number representing the action to execute.
         * \throw std::runtime_error if the actionID exceeds nbActions - 1.
         */
        void doAction(uint64_t actionID) override {
            // Perform the action (not relevant for this environment).
        }

        /**
         * \brief Implementation of the getDataSources method for myNewEnv.
         *
         * \return A vector of references to the fake data sources.
         */
        std::vector<std::reference_wrapper<const Data::DataHandler>> getDataSources() override {
            return dataSources;
        }

        /**
         * \brief Implementation of the getScore method for myNewEnv.
         *
         * \return The current score for the myNewEnv.
         */
        double getScore() const override {
            // Return a fixed score for simplicity.
            return 0.0;
        }

        /**
         * \brief Implementation of the isTerminal method for myNewEnv.
         *
         * \return A boolean indicating whether the myNewEnv has reached a terminal state.
         */
        bool isTerminal() const override {
            // Return false for simplicity.
            return false;
        }
    };
}; // namespace Learn

#endif
