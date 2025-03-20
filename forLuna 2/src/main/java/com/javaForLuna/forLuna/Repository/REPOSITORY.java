package com.javaForLuna.forLuna.Repository;

import com.javaForLuna.forLuna.entity.detailsForMoon;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface REPOSITORY extends MongoRepository<detailsForMoon, String> {
}
