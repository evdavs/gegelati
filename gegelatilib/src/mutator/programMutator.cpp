/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2019 - 2020) :
 *
 * Karol Desnos <kdesnos@insa-rennes.fr> (2019 - 2020)
 * Nicolas Sourbier <nsourbie@insa-rennes.fr> (2020)
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

#include "mutator/programMutator.h"
#include "mutator/mutationParameters.h"
#include "mutator/rng.h"

void Mutator::ProgramMutator::initRandomProgram(
    Program::Program& p, const MutationParameters& params, Mutator::RNG& rng)
{
    // Empty the program
    while (p.getNbLines() > 0) {
        p.removeLine(0);
    }

    // insert random constants in the program
    Data::Constant c_value;
    for (int i = 0; i < p.getEnvironment().getNbConstant(); i++) {
        c_value = {
            rng.getInt32(params.prog.minConstValue, params.prog.maxConstValue)};
        p.getConstantHandler().setDataAt(typeid(Data::Constant), i, c_value);
    }

    // Select the number of line randomly
    const uint64_t nbLine = rng.getUnsignedInt64(1, params.prog.maxProgramSize);
    // Insert them
    while (p.getNbLines() < nbLine) {
        insertRandomLine(p, rng); 
    }

    // Identify Introns
    p.identifyIntrons();
}

bool Mutator::ProgramMutator::deleteRandomLine(Program::Program& p,
                                               Mutator::RNG& rng)
{
    // Line cannot be removed from a program with a single line.
    if (p.getNbLines() <= 1) {
        return false;
    }

    uint64_t lineIndex = rng.getUnsignedInt64(0, p.getNbLines() - 1);
    p.removeLine(lineIndex);
    return true;
}

void Mutator::ProgramMutator::insertRandomLine(Program::Program& p,
                                               Mutator::RNG& rng)
{
    uint64_t lineIndex = rng.getUnsignedInt64(0, p.getNbLines());
    Program::Line& line = p.addNewLine(lineIndex);
    Mutator::LineMutator::initRandomCorrectLine(line, rng);
}

bool Mutator::ProgramMutator::swapRandomLines(Program::Program& p,
                                              Mutator::RNG& rng)
{
    if (p.getNbLines() < 2) {
        return false;
    }
    // Select two distinct random index.
    const uint64_t lineIndex0 = rng.getUnsignedInt64(0, p.getNbLines() - 1);
    uint64_t lineIndex1 = rng.getUnsignedInt64(0, p.getNbLines() - 2);
    lineIndex1 += (lineIndex1 >= lineIndex0) ? 1 : 0;

    p.swapLines(lineIndex0, lineIndex1);

    return true;
}

bool Mutator::ProgramMutator::alterRandomLine(Program::Program& p,
                                              Mutator::RNG& rng)
{
    if (p.getNbLines() < 1) {
        return false;
    }
    // Select a random index.
    const uint64_t lineIndex = rng.getUnsignedInt64(0, p.getNbLines() - 1);
    Mutator::LineMutator::alterCorrectLine(p.getLine(lineIndex), rng);
    return true;
}

bool Mutator::ProgramMutator::alterRandomConstant(
    Program::Program& p, const MutationParameters& params, Mutator::RNG& rng)
{
    const uint64_t constant_idx =
        rng.getUnsignedInt64(0, p.getEnvironment().getNbConstant() - 1);
    p.getConstantHandler().setDataAt(
        typeid(Data::Constant), constant_idx,
        {rng.getInt32(params.prog.minConstValue, params.prog.maxConstValue)});
    return true;
}

bool Mutator::ProgramMutator::mutateProgram(Program::Program& p,
                                            const MutationParameters& params,
                                            Mutator::RNG& rng)
{
    bool anyMutation = false;
    if (p.getNbLines() > 1 && rng.getDouble(0.0, 1.0) < params.prog.pDelete) {
        anyMutation = true;
        deleteRandomLine(p, rng);
    }

    if (p.getNbLines() < params.prog.maxProgramSize &&
        rng.getDouble(0.0, 1.0) < params.prog.pAdd) {
        anyMutation = true;
        insertRandomLine(p, rng);
    }

    if (rng.getDouble(0.0, 1.0) < params.prog.pMutate) {
        anyMutation = true;
        alterRandomLine(p, rng);
    }

    if (rng.getDouble(0.0, 1.0) < params.prog.pSwap) {
        anyMutation = true;
        swapRandomLines(p, rng);
    }

    // mutate the programs constants if they exists
    if (p.getEnvironment().getNbConstant() > 0 &&
        rng.getDouble(0.0, 1.0) < params.prog.pConstantMutation) {
        anyMutation = true;
        alterRandomConstant(p, params, rng);
    }

    // Identify introns
    if (anyMutation) {
        p.identifyIntrons();
    }

    return anyMutation;
}
