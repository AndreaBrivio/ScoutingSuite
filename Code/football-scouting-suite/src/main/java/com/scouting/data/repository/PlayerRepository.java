package com.scouting.data.repository;

import com.scouting.data.model.Player;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.JpaSpecificationExecutor;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface PlayerRepository extends JpaRepository<Player, Long>, JpaSpecificationExecutor<Player> {
    List<Player> findByAgeBetween(Integer minAge, Integer maxAge);
    List<Player> findByPosition(String position);
    List<Player> findBySquad(String squad);
    List<Player> findByNation(String nation);
    List<Player> findByCompetition(String competition);
}