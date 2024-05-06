/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2020) :
 *
 * Pierre-Yves Le Rolland-Raumer <plerolla@insa-rennes.fr> (2020)
 *
 * GEGELATI is an open-source reinforcement learning framework for training
 * artificial intelligence based on Tangled Program Graphs (TPGs).
 *
 * This software is governed by the CeCILL-C license under French law and
 * abiding by the rules of distribution of free software. You can use,
 * modify and/ or redistribute the software under the terms of the CeCILL-C
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty and the software's author, the holder of the
 * economic rights, and the successive licensors have only limited
 * liability.
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading, using, modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean that it is complicated to manipulate, and that also
 * therefore means that it is reserved for developers and experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or
 * data to be ensured and, more generally, to use and operate it in the
 * same conditions as regards security.
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL-C license and that you accept its terms.
 */

#ifndef JOB_H
#define JOB_H

#include <cstdint>
#include <vector>

#include "tpg/tpgVertex.h"

struct {
    double angle;
    double velocity;
}typedef stateEOE;

namespace Learn {
    /**
     * \brief This class embeds roots for the simulations.
     *
     * The goal of the Job is to contain one root, so that each job
     * will match with one simulation/evaluation. A basic learning agent will
     * embed one root per job to do as many simulations as there are roots.
     */
    class Job
    {
      protected:
        /**
         * The root contained in the job.
         */
        const TPG::TPGVertex* root;

        /**
         * Index associated to this job.
         */
        const uint64_t idx;

        /**
         * Seed that will be used to randomize archive.
         */
        const uint64_t archiveSeed;

        /**
         * Vector of previous end of state values
         */
        std::vector<stateEOE> vecStateEOE;



      public:
        /// Deleted default constructor.
        Job() = delete;

        /**
         * \brief Constructor enabling storing elements in the job so that the
         * Learning Agents will be able to use them later.
         *
         * @param[in] root The root that will be encapsulated into the job.
         * @param[in] archiveSeed The archive seed that will be used with this
         * job.
         * @param[in] idx The index of this job.
         */
        Job(const TPG::TPGVertex* root, uint64_t archiveSeed = 0,
            uint64_t idx = 0)
            : root(root), archiveSeed(archiveSeed), idx(idx)
        {
        }

        /// Default virtual destructor
        virtual ~Job() = default;

        /**
         * \brief Getter of index.
         *
         * @return The index of the job.
         */
        uint64_t getIdx() const;

        /**
         * \brief Getter of archiveSeed.
         *
         * @return The archive seed of the job.
         */
        uint64_t getArchiveSeed() const;

        /**
         * \brief Getter of the root.
         *
         * @return The root embedded by the job.
         */
        virtual const TPG::TPGVertex* getRoot() const;
    };
} // namespace Learn

#endif
